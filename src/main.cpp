#include "xtensor/xbuilder.hpp"
#include "xtensor/xslice.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <argparse/argparse.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <istream>
#include <optional>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
namespace fs = std::filesystem;

int main( int argc, char * argv[] )
{
    argparse::ArgumentParser program( "mfpt_processor" );

    program.add_argument( "trajectory_folder" ).help( "" );
    program.add_argument( "--n_disc" ).help( "" ).default_value( 100 ).scan<'i', int>();
    program.add_argument( "--min_samples" ).help( "" ).default_value( 10 ).scan<'i', int>();

    try
    {
        program.parse_args( argc, argv );
    }
    catch( const std::runtime_error & err )
    {
        fmt::print( stderr, "{}\n{}", err.what(), fmt::streamed( program ) );
        return 1;
    }

    fs::path trajectory_folder = program.get<std::string>( "trajectory_folder" );

    int N_DISC      = program.get<int>( "n_disc" );
    int MIN_SAMPLES = program.get<int>( "min_samples" );

    xt::xtensor<double, 1> order_param = xt::linspace( -1.0, 0.9, N_DISC );

    for( const auto & entry : fs::directory_iterator( trajectory_folder ) )
    {
        fmt::print( "Processing {}\n", entry.path().string() );

        int N_TRAJECTORIES = std::distance( fs::directory_iterator( trajectory_folder ), fs::directory_iterator{} ) + 1;
        xt::xtensor<double, 2> order_param_passage_times = xt::zeros<double>( { N_TRAJECTORIES, N_DISC } );
        int idx_traj                                     = 0;

        fmt::print( "Found {} trajectories\n", N_TRAJECTORIES );
        for( const auto & traj_file : fs::directory_iterator( entry ) )
        {

            if( traj_file.path().extension() != ".npy" )
            {
                fmt::print( ".... Skipping {}\n", traj_file.path().string() );
                continue;
            }

            fmt::print( ".... {}\n", traj_file.path().string() );

            auto data_current       = xt::load_npy<double>( traj_file.path().string() );
            int idx_max_order_param = 0;
            auto t0                 = data_current( 0, 0 );

            for( int irow = 0; irow < data_current.shape( 0 ); irow++ )
            {
                auto row  = xt::view( data_current, irow, xt::all() );
                auto t    = row[0] - t0;
                auto o    = row[-1];
                auto omax = order_param[idx_max_order_param];

                //  Is the current order param higher than the current max?
                //  If yes, we add the time to the passage times array
                if( o > omax )
                {
                    auto idx_new_max = xt::argmax( order_param > o )[0];

                    if( idx_new_max == 0 )
                    {
                        idx_new_max = N_DISC;

                        auto v = xt::view(
                            order_param_passage_times, idx_traj, xt::range( idx_max_order_param, idx_new_max ) )
                            = -1;
                        break;
                    }

                    xt::view( order_param_passage_times, idx_traj, xt::range( idx_max_order_param, idx_new_max ) ) += t;
                    idx_max_order_param = idx_new_max;
                }

                if( o >= xt::amax( order_param )() )
                {
                    break;
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
            auto not_nan_times = xt::filter( times, times > 0 );
            n_samples[idx_o]   = not_nan_times.shape( 0 );

            fmt::print( "idx_o {}\n", idx_o );
            fmt::print( "n_samples {}\n", n_samples[idx_o] );

            if( n_samples[idx_o] == 0 )
            {
                continue;
            }
            mean_passage_times[idx_o] = xt::mean( not_nan_times )();
            fmt::print( "mean_passage_times {}\n", mean_passage_times[idx_o] );

            std_passage_times[idx_o] = xt::stddev( not_nan_times )() / std::sqrt( n_samples[idx_o] );
        }

        std::cout << mean_passage_times[2] << "\n";

        order_param        = xt::filter( order_param, n_samples >= MIN_SAMPLES );
        mean_passage_times = xt::filter( mean_passage_times, n_samples >= MIN_SAMPLES );
        std_passage_times  = xt::filter( std_passage_times, n_samples >= MIN_SAMPLES );
        n_samples          = xt::filter( n_samples, n_samples >= MIN_SAMPLES );

        std::ofstream outfile;
        outfile.open( entry.path() / fs::path( "mean_times.txt" ) );

        xt::xtensor<double, 2> res
            = xt::hstack( xt::xtuple( order_param, mean_passage_times, std_passage_times, n_samples ) );

        // std::cout << res << "\n";
        xt::dump_csv( outfile, res );
        // mean_passage_times = xt::filter( mean_passage_times, n_samples >= MIN_SAMPLES );

        outfile.close();
    }
}