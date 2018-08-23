import subprocess
import os
import argparse
import sys

def get_folder_paths(data_direc):
    folder_paths = {}
    for mc in os.listdir(data_direc):
        assert(mc[:3] == 'bkg' or mc[:6] == 'signal'), 'strange folder found in data_direc: ' + mc

        root_files = [f for f in os.listdir(data_direc + mc) if f[-5:] == '.root' and f != 'driver.root']
        assert(len(root_files) == 1), str(len(root_files)) + ' root file(s) found in ' + mc
        folder_paths[mc] = data_direc + mc + '/' + root_files[0]

    return folder_paths

def run_conversions(folder_paths, options):
    to_convert = folder_paths.keys() if options.mcs is None else options.mcs.split(' ')

    for mc in to_convert:
        command = 'python ../helpers/root_to_df.py -r ' + folder_paths[mc] + ' -o ' + options.output + mc + options.extension + '.csv' + ' -c ' + mc
        print 'cmd:', command
        subprocess.call(command, shell=True)

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="""Collect .root MCs and convert them to csvs.""")

    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Data directory. All folders inside this directory must start with 'bkg' or 'signal'.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory. The csvs will be saved in this folder.')
    parser.add_argument('-mcs', '--mcs', type=str, required=False, default=None,
                        help="""Specific directories in the input to be converted to csvs. For instance,
                        '-mcs "bkg_Wtllat bkg_stl"' will only convert the root files in directories [input]bkg_stl and [input]bkg_stl
                        Default: convert all files in the input directory.""")
    parser.add_argument('-ext', '--extension', type=str, required=False, default='',
                        help="""Extensions to the names of the csvs created. For instance, 
                        passing '-ext test_' will save all csvs as 'test_[direc_name].csv'. Default: None """)

    options = parser.parse_args(argv)
    return options

if __name__ == '__main__':
    options = parse_arguments(sys.argv[1:])
    assert(options.input[-1] == '/'), 'input directory should end in /'
    assert(options.output[-1] == '/'), 'output directory should end in /'
    folder_paths = get_folder_paths(options.input)
    run_conversions(folder_paths, options)


##python acquire.py -i ../../WWW/Analysis/Output/ML/ -o ../data/

