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
    'dense_u32': ('std::uint32_t', 'u32', 'uint32'),
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
    'sparse_u32': ('std::uint32_t', 'u32', 'uint32'),
    'sparse_u64': ('std::uint64_t', 'u64', 'uint64'),
    'sparse_c64': ('std::complex<float>', 'c64', 'complex64'),
    'sparse_c128': ('std::complex<double>', 'c128', 'complex128'),
    'sparse_c256': ('std::complex<__float128>', 'c256', 'complex256')}

NUMPY_ARRAY_TYPES = list(NUMPY_ARRAY_TYPES_TO_CPP.keys())
NUMPY_SCALAR_TYPES = list(set([v[2] for v in NUMPY_ARRAY_TYPES_TO_CPP.values()]))
MATCHES_TOKEN = "npe_matches"
ARG_TOKEN = "npe_arg"
DEFAULT_ARG_TOKEN = "npe_default_arg"
BEGIN_CODE_TOKEN = "npe_begin_code"
END_CODE_TOKEN = "npe_end_code"
FUNCTION_TOKEN = "npe_function"
DTYPE_TOKEN = "npe_dtype"
DOC_TOKEN = "npe_doc"
COMMENT_TOKEN = "//"

cpp_command = None  # Name of the command to run for the C preprocessor. Set at input.

LOG_DEBUG = 3
LOG_INFO = 1
LOG_INFO_VERBOSE = 2
LOG_ERROR = 0
verbosity_level = 1  # Integer representing the level of verbosity 0 = only log errors, 1 = normal, 2 = verbose


class ParseError(Exception):
    pass


class SemanticError(Exception):
    pass


def log(log_level, logstr, end='\n', file=sys.stdout):
    if log_level <= verbosity_level:
        print(logstr, end=end, file=file)


def run_cpp(input_str):
    tmpf = tempfile.NamedTemporaryFile(mode="w+", suffix=".cc")
    tmpf.write(input_str)
    tmpf.flush()
    cmd = cpp_command + " -w " + tmpf.name
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
            raise ParseError("Invalid code at line %d:\nCPP Error message:\n%s" %
                             (line_number, err.decode("utf-8")))

        output = output.split('\n')

        parsed_string = ""
        for out_line in output:
            if str(out_line).strip().startswith("#"):
                continue
            else:
                parsed_string += str(out_line) + "\n"

        tokens = parsed_string.split(SPLIT_TOKEN)

        if tokens[-1].strip().startswith("%s()" % stmt_token) and tokens[-1].strip() != "%s()" % stmt_token:
            raise ParseError("Extra tokens after `%s` statement on line %d" % (stmt_token, line_number))
        elif tokens[-1].strip() == "%s()" % stmt_token:
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
    """
    Parse one of several types of tokens
    :param line:
    :param token_list:
    :param line_number:
    :param case_sensitive:
    :return:
    """
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


def parse_stmt_call(token, line, line_number, throw=True):
    """
    Parse a statement of the form <token>(
    """
    try:
        line = parse_token(line.strip(), token, line_number)
        parse_token(line.strip(), '(', line_number)
    except ParseError:
        if throw:
            raise ParseError("Got invalid token at line %d. Expected `%s`" % (line_number, token))
        else:
            return False
    return True


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


class NpeFunction(object):
    def __init__(self):
        self.bound_function_name = ""  # The name of the function we are binding
        self.input_type_groups = []  # Set of allowed types for each group of variables
        self.input_varname_to_group = {}  # Dictionary mapping input variable names to type groups
        self.group_to_input_varname = {}  # Dictionary mapping type groups to input variable names
        self.input_variable_order = []  # List of input variables in order
        self.input_dtypes = {}  # Map of dtype variables. The keys are the names, and the values are a list of tuples
        self.input_variable_meta = {}  # Dictionary mapping variable names to types
        self.binding_source_code = ""  # The source code of the binding
        self.preamble_source_code = ""  # The code that comes before npe_* statements
        self.documentation_string = ""  # Function documentatioo

    def arg_meta_in_order(self):
        """
        Iterate over the arguments and their meta data in the order they were passed in
        :return: An iterator over (argument_name, argument_metadata)
        """
        for arg_name in self.input_variable_order:
            arg_meta = self.input_variable_meta[arg_name]
            yield arg_name, arg_meta

    def parse_matches_statement(self, line, line_number):
        global MATCHES_TOKEN

        line = parse_token(line.strip(), MATCHES_TOKEN, line_number=line_number, case_sensitive=False)
        line = parse_token(line.strip(), '(', line_number=line_number).strip()
        if not line.endswith(')'):
            # TODO: Pretty error message
            raise ParseError("Missing ')' for matches() token at line %d" % line_number)

        return line[:-1]

    def parse_arg_statement(self, line, line_number, is_default):
        global NUMPY_ARRAY_TYPES, MATCHES_TOKEN, ARG_TOKEN

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
            # We're binding a scipy dense or sparse array. Check that the types are valid.
            for type_str in var_types:
                if not is_numpy_type(type_str):
                    # TODO: Pretty error message
                    raise ParseError("Got invalid type, `%s` in %s() at line %d. "
                                     "If multiple types are specified, "
                                     "they must be one of %s" % (type_str, stmt_token, line_number, NUMPY_ARRAY_TYPES))

            if var_name in self.input_varname_to_group:
                # There was a matches() done before the group was created, fix the data structure
                group_idx = self.input_varname_to_group[var_name]
                assert len(self.input_type_groups[group_idx]) == 0
                self.input_type_groups[group_idx] = var_types
            else:
                # This is the first time we're seeing this group
                self.input_type_groups.append(var_types)
                group_id = len(self.input_type_groups) - 1
                self.input_varname_to_group[var_name] = group_id
                self.group_to_input_varname[group_id] = [var_name]
        else:
            assert len(var_types) == 1

            if var_types[0].startswith(MATCHES_TOKEN):
                is_matches = True

                # If the type was enforcing a match on another type, then handle that case
                matches_name = self.parse_matches_statement(var_types[0], line_number=line_number)

                if matches_name in self.input_varname_to_group:
                    group_id = self.input_varname_to_group[matches_name]
                    self.input_varname_to_group[var_name] = group_id
                    if group_id not in self.group_to_input_varname:
                        self.group_to_input_varname[group_id] = []
                    self.group_to_input_varname[group_id].append(var_name)
                else:
                    self.input_type_groups.append([])
                    group_id = len(self.input_type_groups) - 1
                    self.input_varname_to_group[var_name] = group_id
                    self.input_varname_to_group[matches_name] = group_id
                    if group_id not in self.group_to_input_varname:
                        self.group_to_input_varname[group_id] = []
                    self.group_to_input_varname[group_id].append(var_name)
                    self.group_to_input_varname[group_id].append(matches_name)
            else:
                # TODO: Check that type requested is valid? - I'm not sure if we can really do this though.
                pass

        self.input_variable_order.append(var_name)
        self.input_variable_meta[var_name] = VariableMetadata(name=var_name,
                                                              is_matches=is_matches,
                                                              name_or_type=var_types,
                                                              line_number=line_number,
                                                              default_value=var_value)

        return var_name, var_types, var_value

    def parse_dtype_statement(self, line, line_number):
        global DTYPE_TOKEN

        tokens = tokenize_npe_line(DTYPE_TOKEN, line, line_number)

        if len(tokens) < 2:
            raise ParseError("Got too few arguments for `%s` statement at line %d." % (DTYPE_TOKEN, line_number))

        name = tokens[0]
        validate_identifier_name(name)

        types = tokens[1:]

        for type in types:
            if not type in NUMPY_SCALAR_TYPES:
                raise ParseError("%s statement got invalid dtype, `%s`. Expected one of %s." %
                                 (DTYPE_TOKEN, type, NUMPY_SCALAR_TYPES))

        if name not in self.input_dtypes:
            self.input_dtypes[name] = []

        self.input_dtypes[name].append(types)

        return name, types

    def parse_doc_statement(self, line, line_number, skip):
        global DOC_TOKEN
        if not skip:
            return

        tokens = tokenize_npe_line(DOC_TOKEN, line, line_number)

        if len(tokens) == 0:
            raise ParseError("Got %s statement at line %d but no documentation string." % (DOC_TOKEN, line_number))

        if len(tokens) > 1:
            raise ParseError("Got more than one documentation token at in %s statement at line %d. "
                             "Did you forget quotes around the docstring?" % (DOC_TOKEN, line_number))

        self.documentation_string = tokens[0]

        log(LOG_INFO_VERBOSE,
            TermColors.OKGREEN + "NumpyEigen Docstring - %s" % self.documentation_string)

    def parse_begin_code_statement(self, line, line_number):
        global BEGIN_CODE_TOKEN
        line = parse_token(line.strip(), BEGIN_CODE_TOKEN, line_number=line_number, case_sensitive=False)
        line = parse_token(line.strip(), '(', line_number=line_number)
        line = parse_token(line.strip(), ')', line_number=line_number)
        parse_eol_token(line.strip(), line_number=line_number)

    def parse_end_code_statement(self, line, line_number):
        global END_CODE_TOKEN
        line = parse_token(line.strip(), END_CODE_TOKEN, line_number=line_number, case_sensitive=False)
        line = parse_token(line.strip(), '(', line_number=line_number)
        line = parse_token(line.strip(), ')', line_number=line_number)
        parse_eol_token(line.strip(), line_number=line_number)

    def parse_binding_init_statement(self, line, line_number):
        global FUNCTION_TOKEN

        tokens = tokenize_npe_line(BINDING_INIT_TOKEN, line, line_number)
        if len(tokens) > 1:
            raise ParseError(BINDING_INIT_TOKEN + " got extra tokens, %s, at line %d. "
                                                  "Expected only the name of the function." % (
                             tokens[1, :], line_number))
        binding_name = tokens[0]
        validate_identifier_name(binding_name)

        return binding_name

    def frontend_pass(self, lines):
        global ARG_TOKEN, BEGIN_CODE_TOKEN, END_CODE_TOKEN, FUNCTION_TOKEN

        binding_start_line_number = -1

        for line_number in range(len(lines)):
            if len(lines[line_number].strip()) == 0:
                continue
            elif parse_stmt_call(ARG_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                raise ParseError("Got `%s` statement before `%s` at line %d" %
                                 (ARG_TOKEN, BINDING_INIT_TOKEN, line_number + 1))
            elif parse_stmt_call(DEFAULT_ARG_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                raise ParseError("Got `%s` statement before `%s` at line %d" %
                                 (DEFAULT_ARG_TOKEN, BINDING_INIT_TOKEN, line_number + 1))
            elif parse_stmt_call(BEGIN_CODE_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                raise ParseError("Got `%s` statement before `%s` at line %d" %
                                 (BEGIN_CODE_TOKEN, BINDING_INIT_TOKEN, line_number + 1))
            elif parse_stmt_call(END_CODE_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                raise ParseError("Got `%s` statement before `%s` at line %d" %
                                 (END_CODE_TOKEN, BINDING_INIT_TOKEN, line_number + 1))
            elif parse_stmt_call(DTYPE_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                raise ParseError("Got `%s` statement before `%s` at line %d" %
                                 (DTYPE_TOKEN, BINDING_INIT_TOKEN, line_number + 1))
            elif parse_stmt_call(DOC_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                raise ParseError("Got `%s` statement before `%s` at line %d" %
                                 (DOC_TOKEN, BINDING_INIT_TOKEN, line_number + 1))
            elif parse_stmt_call(BINDING_INIT_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                self.bound_function_name = self.parse_binding_init_statement(lines[line_number], line_number=line_number + 1)
                binding_start_line_number = line_number + 1
                break
            else:
                self.preamble_source_code += lines[line_number]
                # raise ParseError("Unexpected tokens at line %d: %s" % (line_number, lines[line_number]))

        if binding_start_line_number < 0:
            raise ParseError("Invalid binding file. Must begin with %s(<function_name>)." % BINDING_INIT_TOKEN)

        log(LOG_INFO_VERBOSE, TermColors.OKGREEN + "NumpyEigen Function: " + TermColors.ENDC + self.bound_function_name)

        code_start_line_number = -1

        parsing_doc = False
        doc_lines = ""
        for line_number in range(binding_start_line_number, len(lines)):
            if parse_stmt_call(ARG_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                var_name, var_types, _ = self.parse_arg_statement(lines[line_number], line_number=line_number + 1,
                                                                  is_default=False)
                log(LOG_INFO_VERBOSE,
                    TermColors.OKGREEN + "NumpyEigen Arg: " + TermColors.ENDC + var_name + " - " + str(var_types))

                self.parse_doc_statement(doc_lines, line_number=line_number + 1, skip=parsing_doc)
                parsing_doc = False
            elif parse_stmt_call(DEFAULT_ARG_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                var_name, var_types, var_value = \
                    self.parse_arg_statement(lines[line_number], line_number=line_number + 1, is_default=True)
                log(LOG_INFO_VERBOSE,
                    TermColors.OKGREEN + "NumpyEigen Default Arg: " + TermColors.ENDC + var_name + " - " +
                    str(var_types) + " - " + str(var_value))

                # If we were parsing a multiline npe_doc, we've now reached the end so parse the whole statement
                self.parse_doc_statement(doc_lines, line_number=line_number + 1, skip=parsing_doc)
                parsing_doc = False

            elif parse_stmt_call(DTYPE_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                dtype_name, dtype_types = self.parse_dtype_statement(lines[line_number], line_number=line_number + 1)
                log(LOG_INFO_VERBOSE,
                    TermColors.OKGREEN + "NumpyEigen DType: " + TermColors.ENDC + dtype_name + " - " +
                    str(dtype_types))

                # If we were parsing a multiline npe_doc, we've now reached the end so parse the whole statement
                self.parse_doc_statement(doc_lines, line_number=line_number + 1, skip=parsing_doc)
                parsing_doc = False

            elif parse_stmt_call(DOC_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                if self.documentation_string != "":
                    raise ParseError(
                        "Multiple `%s` statements for one function at line %d." % (DOC_TOKEN, line_number + 1))

                doc_lines += lines[line_number]
                parsing_doc = True

            elif parse_stmt_call(BEGIN_CODE_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                self.parse_begin_code_statement(lines[line_number], line_number=line_number + 1)
                code_start_line_number = line_number + 1

                # If we were parsing a multiline npe_doc, we've now reached the end so parse the whole statement
                self.parse_doc_statement(doc_lines, line_number=line_number + 1, skip=parsing_doc)
                break

            elif parsing_doc:
                # If we're parsing a multiline doc string, accumulate the line
                doc_lines += lines[line_number]
                continue

            elif len(lines[line_number].strip()) == 0:
                # Ignore newlines and whitespace
                continue

            elif lines[line_number].strip().lower().startswith(COMMENT_TOKEN):
                # Ignore commented lines
                continue

            else:
                raise ParseError("Unexpected tokens at line %d: %s" % (line_number + 1, lines[line_number]))

        if code_start_line_number < 0:
            raise ParseError("Invalid binding file. Must does not contain a %s() statement." % BEGIN_CODE_TOKEN)

        reached_end_token = False
        for line_number in range(code_start_line_number, len(lines)):
            # if lines[line_number].lower().startswith(END_CODE_TOKEN):
            if parse_stmt_call(END_CODE_TOKEN, lines[line_number], line_number=line_number + 1, throw=False):
                self.parse_end_code_statement(lines[line_number], line_number=line_number + 1)
                reached_end_token = True
            elif not reached_end_token:
                self.binding_source_code += lines[line_number]
            elif reached_end_token and len(lines[line_number].strip()) != 0:
                raise ParseError("Expected end of file after %s(). Line %d: %s" %
                                 (END_CODE_TOKEN, line_number, lines[line_number]))

        if not reached_end_token:
            raise ParseError("Unexpected EOF. Binding file must end with a %s() statement." % END_CODE_TOKEN)

    def validate_frontend_output(self):
        global MATCHES_TOKEN

        for var_name, var_meta in self.arg_meta_in_order():
            var_meta = self.input_variable_meta[var_name]
            is_sparse = self.is_sparse_type(var_meta.name_or_type[0])
            for type_name in var_meta.name_or_type:
                if self.is_sparse_type(type_name) != is_sparse:
                    raise SemanticError("Input Variable %s (line %d) has a mix of sparse and dense types."
                                        % (var_name, var_meta.line_number))
            self.input_variable_meta[var_name].is_sparse = is_sparse

        for var_name, var_meta in self.arg_meta_in_order():
            if var_meta.is_matches:
                group_idx = self.input_varname_to_group[var_name]
                matches_name = var_meta.name_or_type[0]
                if len(self.input_type_groups[group_idx]) == 0:
                    raise SemanticError("Input Variable %s (line %d) was declared with type %s but was "
                                        "unmatched with a numpy type." % (var_name, var_meta.line_number, matches_name))
                self.input_variable_meta[var_name].is_sparse = self.is_sparse_type(self.input_type_groups[group_idx][0])