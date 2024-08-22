#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xslice.hpp"
#include <argparse/argparse.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
namespace fs = std::filesystem;

int main( int argc, char * argv[] )
{
    argparse::ArgumentParser program( "mfpt_processor" );

    program.add_argument( "trajectory_folder" ).help( "Trajectory folder" );
    program.add_argument( "-o" ).help( "output file" );
    program.add_argument( "--n_disc" )
        .help( "Number of values to discretise the interval to" )
        .default_value( 200 )
        .scan<'i', int>();
    program.add_argument( "--min_samples" )
        .help( "Minimum number of samples to consider per data point" )
        .default_value( 1 )
        .scan<'i', int>();

    try
    {
        program.parse_args( argc, argv );
    }
    catch( const std::runtime_error & err )
    {
        std::cerr << err.what() << "\n" << program << "\n";
        return 1;
    }

    fs::path trajectory_folder = program.get<std::string>( "trajectory_folder" );
    fs::path output_file       = program.get<std::string>( "o" );

    int N_DISC      = program.get<int>( "n_disc" );
    int MIN_SAMPLES = program.get<int>( "min_samples" );

    xt::xtensor<double, 1> order_param = xt::linspace( -1.0, 1.0, N_DISC );
    xt::xtensor<double, 2> order_param_passage_times;

    int N_TRAJECTORIES = 0;
    for( const auto & f : fs::directory_iterator( trajectory_folder ) )
    {
        if( f.path().extension() != ".npy" )
        {
            continue;
        }
        N_TRAJECTORIES++;
    }

    order_param_passage_times = -xt::ones<double>( { N_TRAJECTORIES, N_DISC } );
    int idx_traj              = 0;

    std::cout << "Found " << N_TRAJECTORIES << " trajectories\n";

    // Iterator over all the trajectories for the current parameter set
    for( const auto & traj_file : fs::directory_iterator( trajectory_folder ) )
    {
        if( traj_file.path().extension() != ".npy" )
        {
            continue;
        }
        std::cout << ".... " << traj_file.path().string() << "\n";

        auto data_current = xt::load_npy<double>( traj_file.path().string() );
        auto t0           = data_current( 0, 0 ); // Time of the first snapshot in the trajectory (in case its not zero)
        int idx_max_order_param = 0;              // idx of the max order parameter

        for( size_t irow = 0; irow < data_current.shape( 0 ); irow++ )
        {
            auto row    = xt::view( data_current, irow, xt::all() ); // One row of the trajectory
            double t    = row[0] - t0;                               // Current time
            double o    = row.periodic( -1 );                        // Current value of the order paramater
            double omax = order_param[idx_max_order_param];          // Running max of the order parameter

            // Is the current order param higher than the current max?
            // If yes, we add the time to the passage times array
            if( o > omax )
            {
                if( xt::count_nonzero( order_param > o )() == 0 )
                {
                    break;
                }

                int idx_new_max = xt::argmax( order_param > o )[0];
                xt::view( order_param_passage_times, idx_traj, xt::range( idx_max_order_param, idx_new_max ) ) = t;
                idx_max_order_param = idx_new_max;
            }
        }

        idx_traj++;
    }

    // Finally, we divide by the number of trajectories to get the mean first passage times
    xt::xtensor<double, 1> mean_passage_times = xt::zeros<double>( { N_DISC } );
    xt::xtensor<double, 1> std_passage_times  = xt::zeros<double>( { N_DISC } );
    xt::xtensor<double, 1> n_samples          = xt::zeros<double>( { N_DISC } );

    for( int idx_o = 0; idx_o < N_DISC; idx_o++ )
    {
        auto times         = xt::view( order_param_passage_times, xt::all(), idx_o );
        auto not_nan_times = xt::filter( times, times >= 0 );
        n_samples[idx_o]   = not_nan_times.shape( 0 );

        if( n_samples[idx_o] == 0 )
        {
            continue;
        }
        mean_passage_times[idx_o] = xt::mean( not_nan_times )();
        std_passage_times[idx_o]
            = xt::stddev( not_nan_times )()
              / std::sqrt( n_samples[idx_o] ); // The error on the mean, hence we divide by sqrt(n_sample)
    }

    auto order_param_filtered        = xt::filter( order_param, n_samples >= MIN_SAMPLES );
    auto mean_passage_times_filtered = xt::filter( mean_passage_times, n_samples >= MIN_SAMPLES );
    auto std_passage_times_filtered  = xt::filter( std_passage_times, n_samples >= MIN_SAMPLES );
    auto n_samples_filtered          = xt::filter( n_samples, n_samples >= MIN_SAMPLES );

    std::ofstream output_file_stream;
    output_file_stream.open( output_file );

    auto tpl = xt::xtuple(
        order_param_filtered, mean_passage_times_filtered, std_passage_times_filtered, n_samples_filtered );

    auto res = xt::transpose( xt::stack( std::move( tpl ) ) );

    xt::dump_csv( output_file_stream, res );
    output_file_stream.close();
}