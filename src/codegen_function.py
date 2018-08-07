import argparse
import itertools
import os
import sys
import tempfile
import subprocess


class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# TODO: Check that your compiler supports __float128
NUMPY_ARRAY_TYPES_TO_CPP = {
    # Dense types
    'dense_f32': ('float', 'f32', 'float32'),
    'dense_f64': ('double', 'f64', 'float64'),
    'dense_f128': ('__float128', 'f128', 'float128'),
    'dense_i8': ('std::int8_t', 'i8', 'int8'),
    'dense_i16': ('std::int16_t', 'i16', 'int16'),
    'dense_i32': ('std::int32_t', 'i32', 'int32'),
    'dense_i64': ('std::int64_t', 'i64', 'int64'),
    'dense_u8': ('std::uint8_t', 'u8', 'uint8'),
    'dense_u16': ('std::uint16_t', 'u16', 'uint16'),
    'dense_u32': ('std::uint32_t', 'u32', 'uing32'),
    'dense_u64': ('std::uint64_t', 'u64', 'uint64'),
    'dense_c64': ('std::complex<float>', 'c64', 'complex64'),
    'dense_c128': ('std::complex<double>', 'c128', 'complex128'),
    'dense_c256': ('std::complex<__float128>', 'c256', 'complex256'),

    # Sparse types
    'sparse_f32': ('float', 'f32', 'float32'),
    'sparse_f64': ('double', 'f64', 'float64'),
    'sparse_f128': ('__float128', 'f128', 'float128'),
    'sparse_i8': ('std::int8_t', 'i8', 'int8'),
    'sparse_i16': ('std::int16_t', 'i16', 'int16'),
    'sparse_i32': ('std::int32_t', 'i32', 'int32'),
    'sparse_i64': ('std::int64_t', 'i64', 'int64'),
    'sparse_u8': ('std::uint8_t', 'u8', 'uint8'),
    'sparse_u16': ('std::uint16_t', 'u16', 'uint16'),
    'sparse_u32': ('std::uint32_t', 'u32', 'uing32'),
    'sparse_u64': ('std::uint64_t', 'u64', 'uint64'),
    'sparse_c64': ('std::complex<float>', 'c64', 'complex64'),
    'sparse_c128': ('std::complex<double>', 'c128', 'complex128'),
    'sparse_c256': ('std::complex<__float128>', 'c256', 'complex256')}

NUMPY_ARRAY_TYPES = list(NUMPY_ARRAY_TYPES_TO_CPP.keys())
MATCHES_TOKEN = "npe_matches"
ARG_TOKEN = "npe_arg"
DEFAULT_ARG_TOKEN = "npe_default_arg"
BEGIN_CODE_TOKEN = "npe_begin_code"
END_CODE_TOKEN = "npe_end_code"
BINDING_INIT_TOKEN = "npe_function"
COMMENT_TOKEN = "//"

CPP_COMMAND = None  # Name of the command to run for the C preprocessor. Set at input.

# TODO: More than one function per file
# TODO: Refactor the frontend as a module that can be run on a file
input_file_name = ""  # The name of the input file. Right now we enforce the one function, one file rule
bound_function_name = ""  # The name of the function we are binding
input_type_groups = []  # Set of allowed types for each group of variables
input_varname_to_group = {}  # Dictionary mapping input variable names to type groups
group_to_input_varname = {}  # Dictionary mapping type groups to input variable names
input_variable_order = []  # List of input variables in order
input_variable_meta = {}  # Dictionary mapping variable names to types
binding_source_code = ""  # The source code of the binding
preamble_source_code = ""  # The code that comes before npe_* statements


class ParseError(Exception):
    pass


class SemanticError(Exception):
    pass


class VariableMetadata(object):
    def __init__(self, name, is_matches, name_or_type, line_number, default_value):
        self.name = name
        self.is_matches = is_matches
        self.name_or_type = name_or_type
        self.line_number = line_number
        self.is_sparse = False
        self.default_value = default_value

    def __repr__(self):
        return str(self.__dict__)


def run_cpp(input_str):
    tmpf = tempfile.NamedTemporaryFile(mode="w+")
    tmpf.write(input_str)
    tmpf.flush()
    cmd = CPP_COMMAND + " -w " + tmpf.name
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    tmpf.close()
    return output.decode('utf-8'), err


def tokenize_npe_line(stmt_token, line, line_number):
    MAX_ITERS = 64
    SPLIT_TOKEN = "__NPE_SPLIT_NL__"
    cpp_str = "#define %s(arg, ...) arg %s %s(__VA_ARGS__)" % (stmt_token, SPLIT_TOKEN, stmt_token)

    cpp_input_str = cpp_str + "\n" + line + "\n"

    exited_before_max_iters = False

    for i in range(MAX_ITERS):
        output, err = run_cpp(cpp_input_str)

        if err:
            raise ParseError("Invalid code at line %d:\n%s" % (line_number, line))

        output = output.split('\n')

        parsed_string = ""
        for out_line in output:
            if str(out_line).strip().startswith("#"):
                continue
            elif not out_line.strip():
                continue
            else:
                parsed_string += str(out_line) + "\n"

        tokens = parsed_string.split(SPLIT_TOKEN)

        if tokens[-1].strip() == "%s()" % stmt_token:
            exited_before_max_iters = True
            tokens.pop()
            break

        cpp_input_str = cpp_str + "\n" + parsed_string

    if not exited_before_max_iters:
        raise ParseError("Reached token parser maximum recursion depth (%d) at line %d" % (MAX_ITERS, line_number))

    tokens = [s.strip() for s in tokens]

    return tokens


def validate_identifier_name(var_name):
    # TODO: Validate identifier name
    pass


def is_numpy_type(typestr):
    global NUMPY_ARRAY_TYPES
    return typestr.lower() in NUMPY_ARRAY_TYPES


def parse_token(line, token, line_number, case_sensitive=True):
    check_line = line if case_sensitive else line.lower()
    check_token = token if case_sensitive else token.lower()
    if not check_line.startswith(check_token):
        # TODO: Pretty error message
        raise ParseError("Missing '%s' at line %d" % (token, line_number))

    return line[len(token):]


def parse_string_token(line, line_number):
    if not line.startswith('"'):
        # TODO: Pretty error message
        raise ParseError("Invalid string token at line %d" % line_number)

    # Handle escaped strings
    str_token = ""
    while True:
        idx = line.find('"', 1)
        if idx < 0:
            raise ParseError("Invalid string token at line %d" % line_number)
        if line[idx-1] == "\\":  # escaped
            str_token += line[1:idx-1] + line[idx]
            line = line[idx:]
        else:
            str_token += line[1:idx]
            break
    return str_token, line[idx+1:]


def parse_eol_token(line, line_number):
    if len(line.strip()) != 0:
        # TODO: Pretty error message
        raise ParseError("Expected end-of-line after ')' token on line %d. Got '%s'" % (line_number, line.strip()))
    return line.strip()


def parse_one_of_tokens(line, token_list, line_number, case_sensitive=True):
    success = False
    ret_token, ret = None, None
    for t in token_list:
        try:
            ret = parse_token(line, t, line_number, case_sensitive=case_sensitive)
            ret_token = t
            success = True
            break
        except ParseError:
            continue
    if not success:
        # TODO: Pretty error message
        raise ParseError("Expected one of %s at line %d" % (token_list, line_number))

    return ret_token, ret


def parse_matches_statement(line, line_number):
    global MATCHES_TOKEN

    line = parse_token(line.strip(), MATCHES_TOKEN, line_number=line_number, case_sensitive=False)
    line = parse_token(line.strip(), '(', line_number=line_number).strip()
    if not line.endswith(')'):
        # TODO: Pretty error message
        raise ParseError("Missing ')' for matches() token at line %d" % line_number)

    return line[:-1]


def parse_arg_statement(line, line_number, is_default):
    global NUMPY_ARRAY_TYPES, MATCHES_TOKEN, ARG_TOKEN
    global input_type_groups, input_varname_to_group, group_to_input_varname

    stmt_token = DEFAULT_ARG_TOKEN if is_default else ARG_TOKEN

    tokens = tokenize_npe_line(stmt_token, line.strip(), line_number)

    var_name = tokens[0]
    var_types = tokens[1:]

    validate_identifier_name(var_name)

    var_value = var_types.pop() if is_default else None

    is_matches = False

    if len(var_types) == 0:
        # TODO: Pretty error message
        raise ParseError('%s("%s") got no type arguments' % (stmt_token, var_name))
    elif len(var_types) > 1 or (len(var_types) == 1 and is_numpy_type(var_types[0])):
        # We're binding a scipy dense or array. Check that the types are valid.
        for type_str in var_types:
            if not is_numpy_type(type_str):
                # TODO: Pretty error message
                raise ParseError("Got invalid type, `%s` in %s() at line %d. "
                                 "If multiple types are specified, "
                                 "they must be one of %s" % (type_str, stmt_token, line_number, NUMPY_ARRAY_TYPES))

        if var_name in input_varname_to_group:
            # There was a matches() done before the group was created, fix the data structure
            group_idx = input_varname_to_group[var_name]
            assert len(input_type_groups[group_idx]) == 0
            input_type_groups[group_idx] = var_types
        else:
            # This is the first time we're seeing this group
            input_type_groups.append(var_types)
            group_id = len(input_type_groups) - 1
            input_varname_to_group[var_name] = group_id
            group_to_input_varname[group_id] = [var_name]
    else:
        assert len(var_types) == 1

        if var_types[0].startswith(MATCHES_TOKEN):
            is_matches = True

            # If the type was enforcing a match on another type, then handle that case
            matches_name = parse_matches_statement(var_types[0], line_number=line_number)

            if matches_name in input_varname_to_group:
                group_id = input_varname_to_group[matches_name]
                input_varname_to_group[var_name] = group_id
                if group_id not in group_to_input_varname:
                    group_to_input_varname[group_id] = []
                group_to_input_varname[group_id].append(var_name)
            else:
                input_type_groups.append([])
                group_id = len(input_type_groups) - 1
                input_varname_to_group[var_name] = group_id
                input_varname_to_group[matches_name] = group_id
                if group_id not in group_to_input_varname:
                    group_to_input_varname[group_id] = []
                group_to_input_varname[group_id].append(var_name)
                group_to_input_varname[group_id].append(matches_name)
        else:
            # TODO: Check that type requested is valid? - I'm not sure if we can really do this though.
            pass

    input_variable_order.append(var_name)
    input_variable_meta[var_name] = VariableMetadata(name=var_name,
                                                     is_matches=is_matches,
                                                     name_or_type=var_types,
                                                     line_number=line_number,
                                                     default_value=var_value)

    return var_name, var_types, var_value


def parse_begin_code_statement(line, line_number):
    global BEGIN_CODE_TOKEN
    line = parse_token(line.strip(), BEGIN_CODE_TOKEN, line_number=line_number, case_sensitive=False)
    line = parse_token(line.strip(), '(', line_number=line_number)
    line = parse_token(line.strip(), ')', line_number=line_number)
    parse_eol_token(line.strip(), line_number=line_number)


def parse_end_code_statement(line, line_number):
    global END_CODE_TOKEN
    line = parse_token(line.strip(), END_CODE_TOKEN, line_number=line_number, case_sensitive=False)
    line = parse_token(line.strip(), '(', line_number=line_number)
    line = parse_token(line.strip(), ')', line_number=line_number)
    parse_eol_token(line.strip(), line_number=line_number)


def parse_binding_init_statement(line, line_number):
    global BINDING_INIT_TOKEN

    tokens = tokenize_npe_line(BINDING_INIT_TOKEN, line, line_number)
    if len(tokens) > 1:
        raise ParseError(BINDING_INIT_TOKEN + " got extra tokens, %s, at line %d. "
                                              "Expected only the name of the function." % (tokens[1, :], line_number))
    binding_name = tokens[0]
    validate_identifier_name(binding_name)

    return binding_name


def frontend_pass(lines):
    global ARG_TOKEN, BEGIN_CODE_TOKEN, END_CODE_TOKEN, BINDING_INIT_TOKEN
    global binding_source_code, bound_function_name, preamble_source_code

    binding_start_line_number = -1

    for line_number in range(len(lines)):
        if len(lines[line_number].strip()) == 0:
            continue
        elif lines[line_number].strip().lower().startswith(ARG_TOKEN):
            raise ParseError("Got `%s` statement before `%s` at line %d" %
                             (ARG_TOKEN, BINDING_INIT_TOKEN, line_number+1))
        elif lines[line_number].strip().lower().startswith(DEFAULT_ARG_TOKEN):
            raise ParseError("Got `%s` statement before `%s` at line %d" %
                             (DEFAULT_ARG_TOKEN, BINDING_INIT_TOKEN, line_number+1))
        elif lines[line_number].strip().lower().startswith(BEGIN_CODE_TOKEN):
            raise ParseError("Got `%s` statement before `%s` at line %d" %
                             (BEGIN_CODE_TOKEN, BINDING_INIT_TOKEN, line_number+1))
        elif lines[line_number].strip().lower().startswith(END_CODE_TOKEN):
            raise ParseError("Got `%s` statement before `%s` at line %d" %
                             (END_CODE_TOKEN, BINDING_INIT_TOKEN, line_number+1))
        elif lines[line_number].strip().lower().startswith(BINDING_INIT_TOKEN):
            bound_function_name = parse_binding_init_statement(lines[line_number], line_number=line_number+1)
            binding_start_line_number = line_number + 1
            break
        else:
            preamble_source_code += lines[line_number]
            # raise ParseError("Unexpected tokens at line %d: %s" % (line_number, lines[line_number]))

    if binding_start_line_number < 0:
        raise ParseError("Invalid binding file. Must begin with %s(<function_name>)." % BINDING_INIT_TOKEN)

    print(TermColors.OKGREEN + "NumpyEigen Function: " + TermColors.ENDC + bound_function_name)

    code_start_line_number = -1

    for line_number in range(binding_start_line_number, len(lines)):
        if lines[line_number].strip().lower().startswith(ARG_TOKEN):
            var_name, var_types, _ = parse_arg_statement(lines[line_number], line_number=line_number+1, is_default=False)
            print(TermColors.OKGREEN + "NumpyEigen Arg: " + TermColors.ENDC + var_name + " - " + str(var_types))
        elif lines[line_number].strip().lower().startswith(DEFAULT_ARG_TOKEN):
            var_name, var_types, var_value = \
                parse_arg_statement(lines[line_number], line_number=line_number+1, is_default=True)
            print(TermColors.OKGREEN + "NumpyEigen Default Arg: " + TermColors.ENDC + var_name + " - " +
                  str(var_types) + " - " + str(var_value))
        elif lines[line_number].strip().lower().startswith(BEGIN_CODE_TOKEN):
            parse_begin_code_statement(lines[line_number], line_number=line_number+1)
            code_start_line_number = line_number + 1
            break
        elif len(lines[line_number].strip()) == 0:
            # Ignore newlines and whitespace
            continue
        elif lines[line_number].strip().lower().startswith(COMMENT_TOKEN):
            # Ignore commented lines
            continue
        else:
            raise ParseError("Unexpected tokens at line %d: %s" % (line_number+1, lines[line_number]))

    if code_start_line_number < 0:
        raise ParseError("Invalid binding file. Must does not contain a %s() statement." % BEGIN_CODE_TOKEN)

    reached_end_token = False
    for line_number in range(code_start_line_number, len(lines)):
        if lines[line_number].lower().startswith(END_CODE_TOKEN):
            parse_end_code_statement(lines[line_number], line_number=line_number+1)
            reached_end_token = True
        elif not reached_end_token:
            binding_source_code += lines[line_number]
        elif reached_end_token and len(lines[line_number].strip()) != 0:
            raise ParseError("Expected end of file after %s(). Line %d: %s" %
                             (END_CODE_TOKEN, line_number, lines[line_number]))

    if not reached_end_token:
        raise ParseError("Unexpected EOF. Binding file must end with a %s() statement." % END_CODE_TOKEN)


def validate_frontend_output():
    global MATCHES_TOKEN
    global input_type_groups, input_variable_meta

    for var_name in input_variable_meta.keys():
        var_meta: VariableMetadata = input_variable_meta[var_name]
        is_sparse = is_sparse_type(var_meta.name_or_type[0])
        for type_name in var_meta.name_or_type:
            if is_sparse_type(type_name) != is_sparse:
                raise SemanticError("Input Variable %s (line %d) has a mix of sparse and dense types."
                                    % (var_name, var_meta.line_number))
        input_variable_meta[var_name].is_sparse = is_sparse

    for var_name in input_variable_meta.keys():
        if var_meta.is_matches:
            group_idx = input_varname_to_group[var_name]
            matches_name = var_meta.name_or_type[0]
            if len(input_type_groups[group_idx]) == 0:
                raise SemanticError("Input Variable %s (line %d) was declared with type %s but was "
                                    "unmatched with a numpy type." % (var_name, var_meta.line_number, matches_name))
            input_variable_meta[var_name].is_sparse = is_sparse_type(input_type_groups[group_idx][0])


PRIVATE_ID_PREFIX = "_NPE_PY_BINDING_"
PRIVATE_NAMESPACE = "npe::detail"
TYPE_STRUCT_PUBLIC_NAME = "npe"
STORAGE_ORDER_ENUM = "StorageOrder"
ALIGNED_ENUM = "Alignment"
TYPE_ID_ENUM = "TypeId"
TYPE_CHAR_ENUM = "NumpyTypeChar"
INDENT = "  "
STORAGE_ORDER_SUFFIXES = ['_cm', '_rm', '_x']
STORAGE_ORDER_SUFFIX_CM = STORAGE_ORDER_SUFFIXES[0]
STORAGE_ORDER_SUFFIX_RM = STORAGE_ORDER_SUFFIXES[1]
STORAGE_ORDER_SUFFIX_XM = STORAGE_ORDER_SUFFIXES[2]
STORAGE_ORDER_CM = "ColMajor"
STORAGE_ORDER_RM = "RowMajor"
STORAGE_ORDER_XM = "NoOrder"
MAP_TYPE_PREFIX = "npe_Map_"
MATRIX_TYPE_PREFIX = "npe_Matrix_"
SCALAR_TYPE_PREFIX = "npe_Scalar_"
FOR_REAL_DEFINE = "__NPE_FOR_REAL__"


def has_numpy_types():
    """
    Returns true if any of the arguments are numpy or scipy types
    :return: true if any of the input arguments are numpy or scipy types
    """
    has = False
    for var_name, var_meta in input_variable_meta.items():
        if is_numpy_type(var_meta.name_or_type[0]) or var_meta.is_matches:
            has = True
            break
    return has


def indent(n: int):
    ret = ""
    for _ in range(n):
        ret += INDENT

    return ret


def type_name_var(var_name):
    return PRIVATE_ID_PREFIX + var_name + "_type_s"


def storage_order_var(var_name):
    return PRIVATE_ID_PREFIX + var_name + "_so"


def type_id_var(var_name):
    return PRIVATE_ID_PREFIX + var_name + "_t_id"


def is_sparse_type(type_name):
    return type_name.startswith("sparse_")


def storage_order_for_suffix(suffix):
    if suffix == STORAGE_ORDER_SUFFIX_CM:
        return PRIVATE_NAMESPACE + "::" + STORAGE_ORDER_ENUM + "::" + STORAGE_ORDER_CM
    elif suffix == STORAGE_ORDER_SUFFIX_RM:
        return PRIVATE_NAMESPACE + "::" + STORAGE_ORDER_ENUM + "::" + STORAGE_ORDER_RM
    elif suffix == STORAGE_ORDER_SUFFIX_XM:
        return PRIVATE_NAMESPACE + "::" + STORAGE_ORDER_ENUM + "::" + STORAGE_ORDER_XM
    else:
        assert False, "major wtf"


def aligned_enum_for_suffix(suffix):
    if suffix == STORAGE_ORDER_SUFFIX_CM or suffix == STORAGE_ORDER_SUFFIX_RM:
        return PRIVATE_NAMESPACE + "::" + ALIGNED_ENUM + "::" + "Aligned"
    elif suffix == STORAGE_ORDER_SUFFIX_XM:
        return PRIVATE_NAMESPACE + "::" + ALIGNED_ENUM + "::" + "Unaligned"
    else:
        assert False, "major wtf"


def type_char_for_numpy_type(np_type):
    return PRIVATE_NAMESPACE + "::" + TYPE_CHAR_ENUM + "::char_" + NUMPY_ARRAY_TYPES_TO_CPP[np_type][1]


def write_flags_getter(out_file, var_name):
    storage_order_var_name = storage_order_var(var_name)
    row_major = PRIVATE_NAMESPACE + "::RowMajor"
    col_major = PRIVATE_NAMESPACE + "::ColMajor"
    no_order = PRIVATE_NAMESPACE + "::NoOrder"
    out_str = INDENT + "const " + PRIVATE_NAMESPACE + "::" + STORAGE_ORDER_ENUM + " " + storage_order_var_name + " = "
    out_str += "(" + var_name + ".flags() & NPY_ARRAY_F_CONTIGUOUS) ? " + col_major + " : "
    out_str += "(" + var_name + ".flags() & NPY_ARRAY_C_CONTIGUOUS ? " + row_major + " : " + no_order + ");\n"
    out_file.write(out_str)


def write_type_id_getter(out_file, var_name):
    out_str = INDENT + "const int " + type_id_var(var_name) + " = "
    type_name = type_name_var(var_name)
    storate_order_name = storage_order_var(var_name)
    is_sparse = PRIVATE_NAMESPACE + "::is_sparse<decltype(" + var_name + ")>::value"
    out_str += PRIVATE_NAMESPACE + "::get_type_id(" + is_sparse + ", " + type_name + ", " + storate_order_name + ");\n"
    out_file.write(out_str)


def write_header(out_file):
    out_file.write("#define " + FOR_REAL_DEFINE + "\n")
    out_file.write("#include <pybind11/pybind11.h>\n")
    out_file.write("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION\n")
    out_file.write("#include <numpy/arrayobject.h>\n")
    out_file.write("#include <pybind11/eigen.h>\n")
    out_file.write("#include <pybind11/numpy.h>\n")
    out_file.write("#include <npe.h>\n")
    out_file.write("#include <sparse_array.h>\n")
    out_file.write("#include <Eigen/Sparse>\n")
    out_file.write(preamble_source_code + "\n")

    write_code_function_definition(out_file)

    # TODO: Use the function name properly
    func_name = "pybind_output_fun_" + os.path.basename(input_file_name).replace(".", "_")
    out_file.write("void %s(pybind11::module& m) {\n" % func_name)
    out_file.write('m.def(')
    out_file.write('"%s"' % bound_function_name)
    out_file.write(", [](")

    # Write the argument list
    for i in range(len(input_variable_order)):
        var_name = input_variable_order[i]
        if var_name in input_varname_to_group:
            # print(var_name, "is_sparse:", input_variable_meta[var_name].is_sparse)
            if input_variable_meta[var_name].is_sparse:
                out_file.write("npe::sparse_array ")
            else:
                out_file.write("pybind11::array ")
            out_file.write(var_name)
        else:
            assert len(input_variable_meta[var_name].name_or_type) == 1
            var_type = input_variable_meta[var_name].name_or_type[0]
            out_file.write(var_type + " ")
            out_file.write(var_name)

        next_token = ", " if i < len(input_variable_order) - 1 else ") {\n"
        out_file.write(next_token)

    # Declare variables used to determine the type at runtime
    for var_name in input_varname_to_group.keys():
        out_file.write(indent(1) + "const char %s = %s.dtype().type();\n" % (type_name_var(var_name), var_name))
        out_file.write(indent(1) + "ssize_t %s_shape_0 = 0;\n" % var_name)
        out_file.write(indent(1) + "ssize_t %s_shape_1 = 0;\n" % var_name)
        out_file.write(indent(1) + "if (%s.ndim() == 1) {\n" % var_name)
        out_file.write(indent(2) + "%s_shape_0 = %s.shape()[0];\n" % (var_name, var_name))
        out_file.write(indent(2) + "%s_shape_1 = %s.shape()[0] == 0 ? 0 : 1;\n" % (var_name, var_name))
        out_file.write(indent(1) + "} else if (%s.ndim() == 2) {\n" % var_name)
        out_file.write(indent(2) + "%s_shape_0 = %s.shape()[0];\n" % (var_name, var_name))
        out_file.write(indent(2) + "%s_shape_1 = %s.shape()[1];\n" % (var_name, var_name))
        out_file.write(indent(1) + "} else if (%s.ndim() > 2) {\n" % var_name)
        out_file.write(indent(2) + "  throw std::invalid_argument(\"Argument " + var_name +
                       " has invalid number of dimensions. Must be 1 or 2.\");\n")
        out_file.write(indent(1) + "}\n")

        write_flags_getter(out_file, var_name)
        write_type_id_getter(out_file, var_name)

    # Ensure the types in each group match
    for group_id in group_to_input_varname.keys():
        group_var_names = group_to_input_varname[group_id]
        group_types = input_type_groups[group_id]
        pretty_group_types = [NUMPY_ARRAY_TYPES_TO_CPP[gt][2] for gt in group_types]

        out_str = "if ("
        for i in range(len(group_types)):
            type_name = group_types[i]
            out_str += type_name_var(group_var_names[0]) + " != " + type_char_for_numpy_type(type_name)
            next_token = " && " if i < len(group_types) - 1 else ") {\n"
            out_str += next_token
        out_str += INDENT + 'throw std::invalid_argument("Invalid type (%s) for argument \'%s\'. ' \
                            'Expected one of %s.");\n' % (pretty_group_types[0], group_var_names[0], pretty_group_types)
        out_str += "}\n"
        out_file.write(out_str)

        assert len(group_var_names) >= 1
        if len(group_var_names) == 1:
            continue

        for i in range(1, len(group_var_names)):
            out_str = "if ("
            out_str += type_id_var(group_var_names[0]) + " != " + type_id_var(group_var_names[i]) + ") {\n"
            out_str += INDENT + 'std::string err_msg = std::string("Invalid type (") + %s::type_to_str(%s) + ' \
                                'std::string(") for argument \'%s\'. Expected it to match argument \'%s\' ' \
                                'which is of type ") + %s::type_to_str(%s) + std::string(".");\n' \
                       % (PRIVATE_NAMESPACE, type_name_var(group_var_names[i]), group_var_names[i],
                          group_var_names[0], PRIVATE_NAMESPACE, type_name_var(group_var_names[0]))
            out_str += INDENT + 'throw std::invalid_argument(err_msg);\n'

            out_str += "}\n"

        out_file.write(out_str)


def write_code_block(out_file, combo):
    out_file.write("{\n")
    for group_id in range(len(combo)):
        type_prefix = combo[group_id][0]
        type_suffix = combo[group_id][1]
        for var_name in group_to_input_varname[group_id]:
            cpp_type = NUMPY_ARRAY_TYPES_TO_CPP[type_prefix][0]
            storage_order_enum = storage_order_for_suffix(type_suffix)
            aligned_enum = aligned_enum_for_suffix(type_suffix)

            out_file.write(indent(2) + "typedef " + cpp_type + " Scalar_" + var_name + ";\n")
            if is_sparse_type(combo[group_id][0]):
                eigen_type = "Eigen::SparseMatrix<" + cpp_type + ", " + \
                               storage_order_enum + ", int>"
                out_file.write("typedef " + eigen_type + " Matrix_%s" % var_name + ";\n")
                out_file.write(indent(2) + "typedef Eigen::MappedSparseMatrix<" + cpp_type + ", " +
                               storage_order_enum + ", int> Map_" + var_name + ";\n")
            else:
                eigen_type = "Eigen::Matrix<" + cpp_type + ", " + "Eigen::Dynamic, " + "Eigen::Dynamic, " + \
                             storage_order_enum + ">"
                out_file.write("typedef " + eigen_type + " Matrix_%s" % var_name + ";\n")
                out_file.write(indent(2) + "typedef Eigen::Map<" + eigen_type + ", " +
                               aligned_enum + "> Map_" + var_name + ";\n")

    call_str = "return callit"
    template_str = "<"
    for var_name, var_meta in input_variable_meta.items():
        if var_name in input_varname_to_group:
            template_str += "Map_" + var_name + ", Matrix_" + var_name + ", Scalar_" + var_name + ","
    template_str = template_str[:-1] + ">("

    call_str = call_str + template_str if has_numpy_types() else call_str + "("

    for var_name, var_meta in input_variable_meta.items():
        if var_name in input_varname_to_group:
            if not var_meta.is_sparse:
                call_str += "Map_" + var_name + "((Scalar_" + var_name + "*) " + var_name + ".data(), " + \
                    var_name + "_shape_0, " + var_name + "_shape_1),"
            else:
                call_str += var_name + ".as_eigen<Matrix_" + var_name + ">(),"
        else:
            call_str += var_name + ","

    call_str = call_str[:-1] + ");\n"
    out_file.write(call_str)
    # out_file.write(binding_source_code + "\n")
    out_file.write("}\n")


def write_code_function_definition(out_file):
    template_str = "template <"
    for arg_name, arg_meta in input_variable_meta.items():
        if arg_name in input_varname_to_group:
            template_str += "typename " + MAP_TYPE_PREFIX + arg_name + ","
            template_str += "typename " + MATRIX_TYPE_PREFIX + arg_name + ","
            template_str += "typename " + SCALAR_TYPE_PREFIX + arg_name + ","
    template_str = template_str[:-1] + ">\n"
    if has_numpy_types():
        out_file.write(template_str)
    out_file.write("static auto callit(")

    argument_str = ""
    for arg_name, arg_meta in input_variable_meta.items():
        if arg_name in input_varname_to_group:
            argument_str += "%s%s %s," % (MAP_TYPE_PREFIX, arg_name, arg_name)
        else:
            arg_meta: VariableMetadata = arg_meta
            argument_str += arg_meta.name_or_type[0] + " " + arg_name + ","
    argument_str = argument_str[:-1] + ") {\n"
    out_file.write(argument_str)
    out_file.write(binding_source_code)
    out_file.write("}\n")


def backend_pass(out_file):
    write_header(out_file)

    expanded_type_groups = [itertools.product(group, STORAGE_ORDER_SUFFIXES) for group in input_type_groups]
    group_combos = itertools.product(*expanded_type_groups)

    branch_count = 0

    if has_numpy_types():
        for combo in group_combos:
            if_or_elseif = "if " if branch_count == 0 else " else if "
            out_str = if_or_elseif + "("

            skip = False
            for group_id in range(len(combo)):
                # Sparse types only have column (csc) and row (csr) matrix types so don't output a branch for unaligned
                if is_sparse_type(combo[group_id][0]) and combo[group_id][1] == STORAGE_ORDER_SUFFIX_XM:
                    skip = True
                    break
                repr_var = group_to_input_varname[group_id][0]
                typename = combo[group_id][0] + combo[group_id][1]
                out_str += type_id_var(repr_var) + " == " + PRIVATE_NAMESPACE + "::" + TYPE_ID_ENUM + "::" + typename
                next_token = " && " if group_id < len(combo)-1 else ")"
                out_str += next_token

            if skip:
                continue

            out_str += " {\n"
            out_file.write(out_str)
            write_code_block(out_file, combo)
            out_file.write("}")
            branch_count += 1
        out_file.write(" else {\n")
        out_file.write(INDENT + 'throw std::invalid_argument("This should never happen but clearly it did. '
                                'File a github issue at https://github.com/fwilliams/numpyeigen");\n')
        out_file.write("}\n")
    else:
        group_combos = list(group_combos)
        assert len(group_combos) == 1, "This should never happen but clearly it did. " \
                                       "File a github issue at https://github.com/fwilliams/numpyeigen"
        for _ in group_combos:
            out_file.write("{\n")
            out_file.write(binding_source_code + "\n")
            out_file.write("}\n")

    out_file.write("\n")
    out_file.write("}")

    arg_list = ""
    for arg_name, arg_meta in input_variable_meta.items():
        arg_list += ", pybind11::arg(\"" + arg_name + "\")"
        arg_list += "=" + arg_meta.default_value if arg_meta.default_value else ""

    out_file.write(arg_list)
    out_file.write(");\n")
    out_file.write("}\n")
    out_file.write("\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("file", type=str)
    arg_parser.add_argument("cpp_cmd", type=str)
    arg_parser.add_argument("-o", "--output", type=str, default="a.out")

    args = arg_parser.parse_args()

    CPP_COMMAND = args.cpp_cmd

    with open(args.file, 'r') as f:
        line_list = f.readlines()

    try:
        input_file_name = args.file
        frontend_pass(line_list)
        validate_frontend_output()
        with open(args.output, 'w+') as outfile:
            backend_pass(outfile)
    except SemanticError as e:
        # TODO: Pretty printer
        print(TermColors.FAIL + TermColors.BOLD + "NumpyEigen Semantic Error: " +
              TermColors.ENDC + TermColors.ENDC + str(e), file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        # TODO: Pretty printer
        print(TermColors.FAIL + TermColors.BOLD + "NumpyEigen Syntax Error: " +
              TermColors.ENDC + TermColors.ENDC + str(e), file=sys.stderr)
        sys.exit(1)
