import argparse
import os


class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


FUNCTION_NAME_PREFIX = "pybind_output_fun_"


def write_module(out_file, module_name, files):
    out_file.write("#include <pybind11/pybind11.h>\n")

    func_names = ["pybind_output_fun_" + os.path.basename(fn).replace(".", "_") for fn in files]

    for fn in func_names:
        out_file.write("void %s(pybind11::module&);\n" % fn)

    out_file.write("PYBIND11_MODULE(%s, m) {\n" % module_name)
    out_file.write("m.doc() = \"TODO: Dodumentation\";\n")

    for fn in func_names:
        out_file.write("%s(m);\n" % fn)

    out_file.write("#ifdef VERSION_INFO\n")
    out_file.write("m.attr(\"__version__\") = VERSION_INFO;\n")
    out_file.write("m.attr(\"__version__\") = \"dev\";\n")
    out_file.write("#endif\n")
    out_file.write("}\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-o", "--output", type=str, default="a.out")
    arg_parser.add_argument("-m", "--module-name", type=str, required=True)
    arg_parser.add_argument("-f", "--files", type=str, nargs="+", required=True)

    args = arg_parser.parse_args()

    print(TermColors.OKGREEN + "NumpyEigen Module:" + TermColors.ENDC + args.module_name)

    with open(args.output, 'w+') as outfile:
        write_module(outfile, args.module_name, args.files)