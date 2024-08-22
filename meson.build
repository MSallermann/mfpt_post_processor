project('mfpt_processor', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20', 'optimization=3'])

add_global_arguments('-Wno-unused-local-typedefs', language : 'cpp')

sources_seldon = [
  'src/main.cpp',
]

exe = executable('mfpt_processor', sources_seldon + 'src/main.cpp',
  install : true,
  dependencies : [dependency('xtensor')]
)