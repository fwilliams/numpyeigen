from __future__ import print_function

import argparse
import itertools
import os
import platform
import re
import subprocess
import sys
import tempfile

"""
Global constants used by NumpyEigen
"""
NUMPY_ARRAY_TYPES_TO_CPP = {
    # Dense types
    'dense_float': ('npy_float', 'float', 'float32'),
    'dense_double': ('npy_double', 'double', 'float64'),
    'dense_longdouble': ('npy_longdouble', 'longdouble', 'float128'),
    'dense_byte': ('npy_byte', 'byte', 'int8'),
    'dense_short': ('npy_short', 'short', 'int16'),
    'dense_int': ('npy_int', 'int', 'int32'),
    'dense_long': ('npy_long', 'long', 'int64'),
    'dense_longlong': ('npy_longlong', 'longlong', 'int128'),
    'dense_ubyte': ('npy_ubyte', 'ubyte', 'uint8'),
    'dense_ushort': ('npy_ushort', 'ushort', 'uint16'),
    'dense_uint': ('npy_int', 'uint', 'uint32'),
    'dense_ulong': ('npy_ulong', 'ulong', 'uint64'),
    'dense_ulonglong': ('npy_ulonglong', 'ulonglong', 'uint128'),
    'dense_c64': ('npy_complex64', 'c64', 'complex64'),
    'dense_c128': ('npy_complex128', 'c128', 'complex128'),
    'dense_c256': ('npy_complex256', 'c256', 'complex256'),
    'dense_bool': ('npy_bool', 'bool', 'bool'),

    # Sparse types
    'sparse_float': ('npy_float', 'float', 'float32'),
    'sparse_double': ('npy_double', 'double', 'float64'),
    'sparse_longdouble': ('npy_longdouble', 'longdouble', 'float128'),
    'sparse_byte': ('npy_byte', 'byte', 'int8'),
    'sparse_short': ('npy_short', 'short', 'int16'),
    'sparse_int': ('npy_int', 'int', 'int32'),
    'sparse_long': ('npy_long', 'long', 'int64'),
    'sparse_longlong': ('npy_longlong', 'longlong', 'int128'),
    'sparse_ubyte': ('npy_ubyte', 'ubyte', 'uint8'),
    'sparse_ushort': ('npy_ushort', 'ushort', 'uint16'),
    'sparse_uint': ('npy_uint', 'uint', 'uint32'),
    'sparse_ulong': ('npy_ulong', 'ulong', 'uint64'),
    'sparse_ulonglong': ('npy_ulonglong', 'ulonglong', 'uint128'),
    'sparse_c64': ('npy_complex64', 'c64', 'complex64'),
    'sparse_c128': ('npy_complex128', 'c128', 'complex128'),
    'sparse_c256': ('npy_complex256', 'c256', 'complex256'),
    'sparse_bool': ('npy_bool', 'bool', 'bool')}

NUMPY_ARRAY_TYPES = list(NUMPY_ARRAY_TYPES_TO_CPP.keys())
NUMPY_SCALAR_TYPES = list(set([v[2] for v in NUMPY_ARRAY_TYPES_TO_CPP.values()]))
MATCHES_TOKEN = "npe_matches"
ARG_TOKEN = "npe_arg"
DEFAULT_ARG_TOKEN = "npe_default_arg"
BEGIN_CODE_TOKEN = "npe_begin_code"
END_CODE_TOKEN = "npe_end_code"
FUNCTION_TOKEN = "npe_function"
DOC_TOKEN = "npe_doc"
COMMENT_TOKEN = "//"
SPARSE_MATCHES_TOKEN = "npe_sparse_like"
DENSE_MATCHES_TOKEN = "npe_dense_like"

LOG_DEBUG = 3
LOG_INFO = 1
LOG_INFO_VERBOSE = 2
LOG_ERROR = 0

"""
Global Variables set at runtime
"""
cpp_command = None  # Name of the command to run for the C preprocessor. Set at input.
cpp_path = None  # Path to the executable
verbosity_level = 1  # Integer representing the level of verbosity 0 = only log errors, 1 = normal, 2 = verbose


class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ParseError(Exception):
    pass


class SemanticError(Exception):
    pass


def log(log_level, logstr, end='\n', file=sys.stdout):
    if log_level <= verbosity_level:
        print(logstr, end=end, file=file)


def tokenize_npe_line(stmt_token, line, line_number, max_iters=64, split_token="__NPE_SPLIT_NL__"):
    """
    Tokenize an NPE statement of the forn STMT(arg1, ..., argN)
    :param stmt_token: The name of the statement to parse.
    :param line: The line to tokenize.
    :param line_number: The line number in the file being parsed.
    :param max_iters: The number of iterations of the C preprocessor to parse everything.
                      This has the side effect of setting the max number of arguments.
    :param split_token: The split token for the C preprocessor to use. Don't change this unless you have a good reason.
    :return:
    """

    def run_cpp(input_str):
        if platform.system() == 'Windows':
            filename = "tmp.cc"
            tmpf = open(filename, "w")
            tmpf.write(input_str)
            tmpf.flush()
            tmpf.close()
        else:
            tmpf = tempfile.NamedTemporaryFile(mode="w+", suffix=".cc")
            tmpf.write(input_str)
            tmpf.flush()
            filename = tmpf.name

        cmd = [filename]
        for c in cpp_command:
            cmd.append(c)

        if platform.system() == 'Windows':
            cmd = [' '.join(cmd)]

        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable=cpp_path)
        cpp_output, cpp_err = p.communicate()

        cpp_err = cpp_err.decode("utf-8")
        cpp_err = re.sub(r'(Microsoft \(R\)).+', '', cpp_err)
        cpp_err = re.sub(r'(Copyright \(C\)).+', '', cpp_err)
        cpp_err = re.sub(r'(' + filename + ')', '', cpp_err)
        cpp_err = cpp_err.strip()

        tmpf.close()
        return cpp_output.decode('utf-8'), cpp_err, filename

    cpp_str = "#define %s(arg, ...) arg %s %s(__VA_ARGS__)" % (stmt_token, split_token, stmt_token)

    cpp_input_str = cpp_str + "\n" + line + "\n"

    exited_before_max_iters = False

    tokens = []
    for i in range(max_iters):
        output, err, filename = run_cpp(cpp_input_str)

        if err:
            raise ParseError("Invalid code at line %d:\nCPP Error message:\n%s" %
                             (line_number, err))

        output = output.split('\n')

        parsed_string = ""
        for out_line in output:
            if str(out_line).strip().startswith("#"):
                continue
            elif str(out_line).strip().startswith("Microsoft (R)"):
                continue
            elif str(out_line).strip().startswith("Copyright (C) Microsoft Corporation"):
                continue
            elif str(out_line).strip() == filename:
                continue
            else:
                parsed_string += str(out_line) + "\n"

        tokens = parsed_string.split(split_token)

        if tokens[-1].strip().startswith("%s()" % stmt_token) and tokens[-1].strip() != "%s()" % stmt_token:
            raise ParseError("Extra tokens after `%s` statement on line %d" % (stmt_token, line_number))
        elif tokens[-1].strip() == "%s()" % stmt_token:
            exited_before_max_iters = True
            tokens.pop()
            break

        cpp_input_str = cpp_str + "\n" + parsed_string

    if not exited_before_max_iters:
        raise ParseError("Reached token parser maximum recursion depth (%d) at line %d" % (max_iters, line_number))

    if len(tokens) == 0:
        raise RuntimeError("This should never happen but clearly it did. "
                           "File a github issue at https://github.com/fwilliams/numpyeigen")

    tokens = [s.strip() for s in tokens]

    return tokens


def validate_identifier_name(var_name):
    """
    Returns True if var_name is a valid C++ identifier, otherwise throws a ParseError
    :param var_name: The name of the identifier to check
    :return: True if var_name is a valid C++ identifier
    """
    # TODO: Validate identifier name
    return var_name


def is_numpy_type(type_name):
    """
    Return True if typestr refers to a NPE type corresponding to a Numpy/Eigen type
    :param type_name: The type string to check
    :return: True if typestr names a Numpy/Eigen type (e.g. dense_f64, sparse_i32)
    """
    return type_name.lower() in NUMPY_ARRAY_TYPES


def is_sparse_type(type_name):
    """
    Return True if typestr refers to a NPE type corresponding to a Numpy/Eigen sparse type
    :param type_name: The type string to check
    :return: True if typestr names a Numpy/Eigen sparse type (e.g. sparse_f64, sparse_i32)
    """
    return is_numpy_type(type_name) and type_name.startswith("sparse_")


def is_dense_type(type_name):
    """
    Return True if typestr refers to a NPE type corresponding to a Numpy/Eigen dense type
    :param type_name: The type string to check
    :return: True if typestr names a Numpy/Eigen dense type (e.g. dense_f64, dense_i32)
    """
    return is_numpy_type(type_name) and type_name.startswith("dense_")


def consume_token(line, token, line_number, case_sensitive=True):
    """
    Consume the token, token, from the input line, ignoring leading whitespace. If the line
    does not start with token, then throw a ParseError
    :param line: The line from which to consume the token
    :param token: The token string to consume
    :param line_number: The line number in the file being read used for error reporting
    :param case_sensitive: Whether parsing is case sensitive or not
    :return: The line with the input token stripped
    """
    check_line = line if case_sensitive else line.lower()
    check_token = token if case_sensitive else token.lower()
    if not check_line.startswith(check_token):
        # TODO: Pretty error message
        raise ParseError("Missing '%s' at line %d" % (token, line_number))

    return line[len(token):]


def consume_eol(line, line_number):
    """
    Consumes whitespace at the end of a line
    :param line: The line from which to consume from
    :param line_number: The number of the line in the file being parsed, used for error reporting
    :return: An empty string on success
    """
    if len(line.strip()) != 0:
        # TODO: Pretty error message
        raise ParseError("Expected end-of-line after ')' token on line %d. Got '%s'" % (line_number, line.strip()))
    return line.strip()


def consume_call_statement(token, line, line_number, throw=True):
    """
    Consume the tokens <token> and <(> from the input line ignoring whitespace
    :param token: The name of the call token
    :param line: The line to check
    :param line_number: The number of the line in the file being read
    :param throw: Whether to throw an exception or simply return False if the line does not start with "token("
    :return: The line with the start tokens consumed or False if throw=False and the line did not start with "token("
    """
    try:
        line = consume_token(line.strip(), token, line_number)
        consume_token(line.strip(), '(', line_number)
    except ParseError:
        if throw:
            raise ParseError("Got invalid token at line %d. Expected `%s`" % (line_number, token))
        else:
            return False
    return line


class NpeFileReader(object):
    def __init__(self, name):
        self.file_name = name
        self.file = open(name, 'r')
        self.line_number = 0
        self.line = ""

    def close(self):
        return self.file.close()

    def readline(self):
        self.line_number += 1
        return self.file.readline()

    def peekline(self):
        if self.file.closed:
            return ""
        pos = self.file.tell()
        line = self.readline()
        self.line_number -= 1
        self.file.seek(pos)
        return line

    def __enter__(self):  # To allow using in 'with' statements
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __next__(self):
        self.line = self.readline()
        if len(self.line) == 0:
            raise StopIteration()
        else:
            return self.line

    def next(self):  # Python 2.7 compatibility
        return self.__next__()

    def __iter__(self):
        return self


class NpeArgument(object):
    def __init__(self, name, is_matches, types, line_number, default_value):
        self.name = name
        self.is_matches = is_matches
        self.scalar_matches_only = False
        self.dense_scalar_matches = False  # If scalar_matches_only is set, determines whether this is from dense_like
        self.types = types
        self.line_number = line_number
        self.is_sparse = False
        self.is_dense = False
        self.default_value = default_value
        self.group = None
        self.matches_name = ""
        self._is_nullable = False

    @property
    def is_numpy_type(self):
        return self.is_sparse or self.is_dense or self.is_matches

    @property
    def is_nullable(self):
        return self.is_matches and self.default_value is not None

    def __repr__(self):
        return str(self.__dict__)


class NpeArgumentGroup(object):
    def __init__(self, types=None, arguments=None):
        self.arguments = arguments if arguments is not None else []
        self.types = types if types is not None else []
        self.meta_group = None  # Other group whose scalar type must match this one

        # Representative argument for this group. This is the one argument that contains scalar types
        self.rep_arg = None

        # The scalar type of an argument in this group must match the scalar type in the other group
        self.group_matches = None
        self.dense_matches = False
        self.sparse_matches = False

        self.id = -1  # Which is my index in the list of groups

        # The NpeMetaArgumentGroup for which this argument belongs to
        self.meta_group = None

    def __repr__(self):
        return str(self.__dict__)


class NpeMetaArgumentGroup(object):
    def __init__(self, groups=None):
        self.groups = groups if groups is not None else []
        self.rep_group = None

    def __repr__(self):
        return str(self.__dict__)


class NpeFunction(object):
    def __init__(self, file_reader):
        self.name = ""  # The name of the function we are binding
        self._argument_groups = []  # Set of allowed types for each group of variables
        self._metagroups = []  # Groups of NpeArgumentGroups which must have matching dtypes
        self._arguments = {}  # Dictionary mapping argument names to NpeArguments
        self.source_code = ""  # The source code of the binding
        self.preamble = ""  # The code that comes before npe_* statements
        self._docstr = ""  # Function documentation

        self._parse(file_reader)
        self._validate()

    @property
    def arguments(self):
        """
        Iterate over the arguments and their meta data in the order they were passed in
        :return: An iterator over (argument_name, argument_metadata)
        """
        for _, arg_meta in self._arguments.items():
            yield arg_meta

    def argument(self, argname):
        return self._arguments[argname]

    @property
    def array_arguments(self):
        for _, arg_meta in self._arguments.items():
            if arg_meta.is_numpy_type:
                yield arg_meta

    @property
    def num_arguments(self):
        return len(self._arguments)

    @property
    def num_argument_groups(self):
        return len(self._argument_groups)

    @property
    def has_array_arguments(self):
        """
        Returns true if any of the arguments are numpy or scipy types
        :return: true if any of the input arguments are numpy or scipy types
        """
        has = False
        for arg in self.arguments:
            if is_numpy_type(arg.types[0]) or arg.is_matches:
                has = True
                break
        return has

    @property
    def argument_groups(self):
        return self._argument_groups

    @property
    def metagroups(self):
        return self._metagroups

    @property
    def docstring(self):
        return self._docstr

    @staticmethod
    def _parse_matches_statement(line, line_number, matches_token_=MATCHES_TOKEN):
        """
        Parse a matches(<type>) statement, returning the the type inside the parentheses
        :param line: The current line being parsed
        :param line_number: The number of the line in the file being parsed
        :return: The name of the type inside the matches statement as a string
        """

        line = consume_token(line.strip(), matches_token_, line_number=line_number, case_sensitive=False)
        line = consume_token(line.strip(), '(', line_number=line_number).strip()
        if not line.endswith(')'):
            raise ParseError("Missing ')' for %s() token at line %d" % (matches_token_, line_number))

        return line[:-1]

    def _parse_arg_statement(self, line, line_number, is_default):
        stmt_token = DEFAULT_ARG_TOKEN if is_default else ARG_TOKEN

        tokens = tokenize_npe_line(stmt_token, line.strip(), line_number)

        arg_name = tokens[0]
        validate_identifier_name(arg_name)

        arg_types = tokens[1:]
        if len(arg_types) == 0:
            raise ParseError('%s("%s") at line %d got no type arguments' % (stmt_token, arg_name, line_number))

        # We allow npe_default_arg(a, npe_matches(b))
        # in which case we want to handle this as a normal matches statement and write out the default arg later
        arg_value = arg_types.pop() if is_default else None
        if arg_value is not None and arg_value.startswith(MATCHES_TOKEN):
            self._parse_matches_statement(arg_value, line_number=line_number)
            arg_types.append(arg_value)

        arg = NpeArgument(name=arg_name,
                          is_matches=False,
                          types=arg_types,
                          line_number=line_number,
                          default_value=arg_value)

        if len(arg_types) == 0:
            # TODO: Pretty error message
            raise ParseError('No types specified for argument %s (line %d)' % (arg_name, line_number))
        elif len(arg_types) > 1 or (len(arg_types) == 1 and is_numpy_type(arg_types[0])):
            # We're binding a scipy dense or sparse array. Check that the types are valid.
            for type_str in arg_types:
                if not is_numpy_type(type_str):
                    # TODO: Pretty error message
                    raise ParseError("Got invalid type, `%s` in %s() at line %d. "
                                     "If multiple types are specified, they must be a valid Numpy or Scipy type. "
                                     "i.e. one of %s" % (type_str, stmt_token, line_number, NUMPY_ARRAY_TYPES))
                arg.is_sparse = is_sparse_type(type_str)
                arg.is_dense = is_dense_type(type_str)
                assert arg.is_sparse != arg.is_dense
        else:
            assert len(arg_types) == 1  # Non NumPy/SciPy arguments can only have one type
            sparse_or_dense_like = arg_types[0].startswith(SPARSE_MATCHES_TOKEN) or \
                                   arg_types[0].startswith(DENSE_MATCHES_TOKEN)
            if arg_types[0].startswith(MATCHES_TOKEN) or sparse_or_dense_like:
                arg.is_matches = True
                arg.scalar_matches_only = sparse_or_dense_like

                # If the type was enforcing a match on another type, then handle that case
                matches_token = MATCHES_TOKEN
                if arg_types[0].startswith(SPARSE_MATCHES_TOKEN):
                    matches_token = SPARSE_MATCHES_TOKEN
                    arg.dense_scalar_matches = False
                    arg.is_dense = False
                    arg.is_sparse = True
                elif arg_types[0].startswith(DENSE_MATCHES_TOKEN):
                    matches_token = DENSE_MATCHES_TOKEN
                    arg.dense_scalar_matches = True
                    arg.is_dense = True
                    arg.is_sparse = False
                arg.matches_name = self._parse_matches_statement(arg_types[0], line_number=line_number,
                                                                 matches_token_=matches_token)
            else:
                # TODO: Check that type requested is valid? - I'm not sure if we can really do this though.
                pass

        self._arguments[arg_name] = arg

        return arg_name, arg_types, arg_value

    def _parse_doc_statement(self, line, line_number, skip):
        if not skip:
            return

        tokens = tokenize_npe_line(DOC_TOKEN, line, line_number)

        if len(tokens) == 0:
            raise ParseError("Got %s statement at line %d but no documentation string." % (DOC_TOKEN, line_number))

        if len(tokens) > 1:
            raise ParseError("Got more than one documentation token at in %s statement at line %d. "
                             "Did you forget quotes around the docstring?" % (DOC_TOKEN, line_number))

        self._docstr = tokens[0]

        log(LOG_INFO_VERBOSE,
            TermColors.OKGREEN + "NumpyEigen Docstring - %s" % self._docstr)

    @staticmethod
    def _parse_npe_function_statement(line, line_number):
        tokens = tokenize_npe_line(FUNCTION_TOKEN, line, line_number)
        if len(tokens) > 1:
            raise ParseError(FUNCTION_TOKEN + " got extra tokens, %s, at line %d. "
                                              "Expected only the name of the function." %
                             (tokens[1, :], line_number))
        binding_name = tokens[0]
        validate_identifier_name(binding_name)

        return binding_name

    @staticmethod
    def _parse_begin_code_statement(line, line_number):
        line = consume_token(line.strip(), BEGIN_CODE_TOKEN, line_number=line_number, case_sensitive=False)
        line = consume_token(line.strip(), '(', line_number=line_number)
        line = consume_token(line.strip(), ')', line_number=line_number)
        consume_eol(line.strip(), line_number=line_number)

    @staticmethod
    def _parse_end_code_statement(line, line_number):
        line = consume_token(line.strip(), END_CODE_TOKEN, line_number=line_number, case_sensitive=False)
        line = consume_token(line.strip(), '(', line_number=line_number)
        line = consume_token(line.strip(), ')', line_number=line_number)
        consume_eol(line.strip(), line_number=line_number)

    def _parse(self, file_reader):
        line = file_reader.readline()
        if consume_call_statement(FUNCTION_TOKEN, line, line_number=file_reader.line_number, throw=False):
            self.name = self._parse_npe_function_statement(line, line_number=file_reader.line_number)
        else:
            raise RuntimeError("This should never happen but clearly it did. "
                               "File a github issue at https://github.com/fwilliams/numpyeigen")

        log(LOG_INFO_VERBOSE, TermColors.OKGREEN + "NumpyEigen Function: " + TermColors.ENDC + self.name)

        found_begin_code_statement = False

        parsing_doc = False
        doc_lines = ""
        for line in file_reader:
            if consume_call_statement(ARG_TOKEN, line, line_number=file_reader.line_number, throw=False):
                var_name, var_types, _ = self._parse_arg_statement(line, line_number=file_reader.line_number,
                                                                   is_default=False)
                log(LOG_INFO_VERBOSE,
                    TermColors.OKGREEN + "NumpyEigen Arg: " + TermColors.ENDC + var_name + " - " + str(var_types))

                self._parse_doc_statement(doc_lines, line_number=file_reader.line_number, skip=parsing_doc)
                parsing_doc = False
            elif consume_call_statement(DEFAULT_ARG_TOKEN, line, line_number=file_reader.line_number, throw=False):
                var_name, var_types, var_value = \
                    self._parse_arg_statement(line, line_number=file_reader.line_number, is_default=True)
                log(LOG_INFO_VERBOSE,
                    TermColors.OKGREEN + "NumpyEigen Default Arg: " + TermColors.ENDC + var_name + " - " +
                    str(var_types) + " - " + str(var_value))

                # If we were parsing a multiline npe_doc, we've now reached the end so parse the whole statement
                self._parse_doc_statement(doc_lines, line_number=file_reader.line_number, skip=parsing_doc)
                parsing_doc = False

            elif consume_call_statement(DOC_TOKEN, line, line_number=file_reader.line_number, throw=False):
                if self._docstr != "":
                    raise ParseError(
                        "Multiple `%s` statements for one function at line %d."
                        % (DOC_TOKEN, file_reader.line_number))

                doc_lines += line
                parsing_doc = True

            elif consume_call_statement(BEGIN_CODE_TOKEN, line, line_number=file_reader.line_number, throw=False):
                self._parse_begin_code_statement(line, line_number=file_reader.line_number)
                found_begin_code_statement = True

                # If we were parsing a multiline npe_doc, we've now reached the end so parse the whole statement
                self._parse_doc_statement(doc_lines, line_number=file_reader.line_number, skip=parsing_doc)
                break

            elif parsing_doc:
                # If we're parsing a multiline doc string, accumulate the line
                doc_lines += line
                continue

            elif len(line.strip()) == 0:
                # Ignore newlines and whitespace
                continue

            elif line.strip().lower().startswith(COMMENT_TOKEN):
                # Ignore commented lines
                continue

            else:
                raise ParseError("Unexpected tokens at line %d: %s" % (file_reader.line_number, line))

        if not found_begin_code_statement:
            raise ParseError("Invalid binding file. Must does not contain a %s() statement." % BEGIN_CODE_TOKEN)

        reached_end_token = False
        for line in file_reader:
            if consume_call_statement(END_CODE_TOKEN, line, line_number=file_reader.line_number, throw=False):
                self._parse_end_code_statement(line, line_number=file_reader.line_number)
                reached_end_token = True
                break
            elif not reached_end_token:
                self.source_code += line

        if not reached_end_token:
            raise ParseError("Unexpected EOF. Binding file must end with a %s() statement." % END_CODE_TOKEN)

    def _validate(self):
        for arg in self.arguments:
            if arg.is_numpy_type:
                if arg.is_matches:  # Argument type is a matches constraint of some kind
                    # Check that the matched argument exists
                    if arg.matches_name not in self._arguments:
                        raise SemanticError("Argument %s is declared with npe_matches(%s) (line %d) but %s is not an "
                                            "argument to the function." %
                                            (arg.name, arg.matches_name, arg.line_number, arg.matches_name))
                    # Check that the matched argument is a Numpy or Scipy type
                    matched_arg = self._arguments[arg.matches_name]
                    if not matched_arg.is_numpy_type:
                        raise SemanticError("Argument %s is declared with npe_matches(%s) (line %d) but %s is not a "
                                            "Numpy or Scipy argument argument." %
                                            (arg.name, arg.matches_name, arg.line_number, arg.matches_name))

                    # Handle npe_matches, npe_sparse/dense_like constraints
                    if not arg.scalar_matches_only:  # Hard matches another argument
                        # Merge the two arguments into an argument group
                        if matched_arg.group is None:
                            grp = NpeArgumentGroup()
                            self._argument_groups.append(grp)
                            grp.id = len(self._argument_groups) - 1
                            matched_arg.group = grp

                        arg.group = matched_arg.group
                    else:  # Dtype matches another argument
                        # Put arguments into seperate groups, and enforce that the dtypes of these groups need to match
                        if arg.group is None:
                            grp = NpeArgumentGroup()
                            self._argument_groups.append(grp)
                            grp.id = len(self._argument_groups) - 1
                            arg.group = grp
                        if matched_arg.group is None:
                            grp = NpeArgumentGroup()
                            self._argument_groups.append(grp)
                            grp.id = len(self._argument_groups) - 1
                            matched_arg.group = grp

                        # Add constraint that dtypes of both groups need to match
                        arg.group.group_matches = matched_arg.group
                        arg.group.dense_matches = arg.dense_scalar_matches
                        arg.group.sparse_matches = not arg.group.dense_matches

                        # The argument which has the dense_like or sparse_like is the representative of this group
                        assert arg.group.rep_arg is None, \
                            "There should never be another representative argument when assigning one."
                        arg.group.rep_arg = arg

                else:  # Argument type is not a matches constraint
                    # Determine if a Numpy/Scipy argument is sparse or dense and check that the user did not
                    # mix sparse and dense types
                    is_sparse = is_sparse_type(arg.types[0])
                    is_dense = is_dense_type(arg.types[0])
                    for type_name in arg.types:
                        if is_sparse_type(type_name) != is_sparse or is_dense_type(type_name) != is_dense:
                            raise SemanticError(
                                "Argument %s (line %d) is declared with a mix of sparse and dense types."
                                % (arg.name, arg.line_number))

                    # Make a new group and put the argument into it
                    if arg.group is None:
                        grp = NpeArgumentGroup()
                        self._argument_groups.append(grp)
                        grp.id = len(self._argument_groups) - 1
                        arg.group = grp
                    assert arg.group is not None
                    assert len(arg.group.types) == 0
                    assert arg.is_dense != arg.is_sparse
                    assert arg.group.rep_arg is None, \
                        "There should never be another representative argument when assigning one."
                    # The argument which has the list of types is the representative of this group
                    arg.group.rep_arg = arg
                    arg.group.types = arg.types

                # Add the argument to its group
                arg.group.arguments.append(arg)

        # Create meta groups for matching arguments
        for grp in self._argument_groups:
            assert grp.rep_arg is not None
            assert grp.rep_arg.is_sparse != grp.rep_arg.is_dense
            for arg in grp.arguments:
                arg.is_dense = grp.rep_arg.is_dense
                arg.is_sparse = grp.rep_arg.is_sparse

            if grp.group_matches is not None:  # This group has a scalar matches constraint to another group
                # Create a MetaGroup containing the two ArgumentGroups
                other_grp = grp.group_matches

                if other_grp.meta_group is None:
                    other_grp.meta_group = NpeMetaArgumentGroup()
                    other_grp.meta_group.groups.append(other_grp)
                    self._metagroups.append(other_grp.meta_group)

                other_grp.meta_group.groups.append(grp)
                grp.meta_group = other_grp.meta_group
            else:  # This group does not have a scalar matches constraint
                # Create MetaGroup contining only this ArgumentGroup
                if grp.meta_group is None:
                    grp.meta_group = NpeMetaArgumentGroup()
                    grp.meta_group.groups.append(grp)
                    self._metagroups.append(grp.meta_group)

            assert grp.meta_group is not None

            # One group in a MetaGroup is a representative. It contains a list of types.
            # It should be impossible for there to be more than one such group.
            if len(grp.types) > 0:
                assert grp.meta_group.rep_group is None, \
                    "There can only be one representative group in a meta group. This should never happen!"
                grp.meta_group.rep_group = grp

        for smg in self._metagroups:
            # scalar_matches cycle
            if smg.rep_group is None:
                cycle_args = ", ".join([grp.rep_arg.name for grp in smg.groups])
                raise SemanticError("The types for arguments %s all reference each other forming a cycle." % cycle_args)

            # Resolve types in each group
            for grp in smg.groups:
                if grp != smg.rep_group:
                    assert grp.meta_group is not None
                    assert grp.dense_matches != grp.sparse_matches
                    if grp.dense_matches:
                        grp.types = [t.replace("sparse_", "dense_") for t in smg.rep_group.types]
                    elif grp.sparse_matches:
                        grp.types = [t.replace("dense_", "sparse_") for t in smg.rep_group.types]

        # Resolve types of all arguments
        for grp in self._argument_groups:
            assert len(grp.arguments) != 0

            # Check that every argument group contains types
            # i.e. no circular npe_matches() arguments
            if len(grp.types) == 0:
                cycle_args = ", ".join([arg.name for arg in grp.arguments])
                raise SemanticError("The types for arguments %s all reference each other forming a npe_matches cycle." % cycle_args)

            # Paranoid double check that all arguments in a group are uniquely sparse or uniquely dense
            # We validate this condition already at parse time so this is an assert
            dense_group = grp.arguments[0].is_dense
            sparse_group = grp.arguments[0].is_sparse
            for arg in grp.arguments:
                # We validate this condition already at parse time so this is an assert
                assert arg.is_dense == dense_group and arg.is_sparse == sparse_group, \
                    "Argument group contains a mix of dense and sparse types. This should never happen!"
                assert arg.is_dense != arg.is_sparse, "An argument is somehow dense and sparse simultaneously. " \
                                                      "This should never happen!"

        # If verbose printing is on, dump all the metadata about this function
        if verbosity_level >= LOG_INFO_VERBOSE:
            logstr = TermColors.OKGREEN + "NumpyEigen Function Metadata:\n" + TermColors.ENDC
            logstr += TermColors.OKGREEN + " Arguments: " + TermColors.ENDC + "%s\n" % [a.name for a in self.arguments]
            logstr += TermColors.OKGREEN + " Groups:\n" + TermColors.ENDC
            for idx, grp in enumerate(self._argument_groups):
                grpargs = [a.name for a in grp.arguments]
                logstr += "  %d: %s\n" % (idx, str(grpargs))
            logstr += TermColors.OKGREEN + " MetaGroups:\n" + TermColors.ENDC
            for metagrp in self._metagroups:
                logstr += "  %s\n" % str([g.id for g in metagrp.groups])
            log(LOG_INFO_VERBOSE, logstr)


class NpeAST(object):
    def __init__(self, file_reader):
        self.input_file_name = file_reader.file_name
        self.children = []
        self._parse(file_reader)

    def _parse(self, file_reader):

        _preamble = ""
        while True:
            line = file_reader.peekline()
            if len(line) == 0:
                file_reader.readline()
                break

            if len(line.strip()) == 0:
                _preamble += file_reader.readline()
                continue
            elif consume_call_statement(ARG_TOKEN, line, line_number=file_reader.line_number + 1, throw=False):
                raise ParseError("Got unexpected `%s`  at line %d" %
                                 (ARG_TOKEN, file_reader.line_number + 1))
            elif consume_call_statement(DEFAULT_ARG_TOKEN, line, line_number=file_reader.line_number + 1, throw=False):
                raise ParseError("Got unexpected `%s`  at line %d" %
                                 (DEFAULT_ARG_TOKEN, file_reader.line_number + 1))
            elif consume_call_statement(BEGIN_CODE_TOKEN, line, line_number=file_reader.line_number + 1, throw=False):
                raise ParseError("Got unexpected `%s`  at line %d" %
                                 (BEGIN_CODE_TOKEN, file_reader.line_number + 1))
            elif consume_call_statement(END_CODE_TOKEN, line, line_number=file_reader.line_number + 1, throw=False):
                raise ParseError("Got unexpected `%s`  at line %d" %
                                 (END_CODE_TOKEN, file_reader.line_number + 1))
            elif consume_call_statement(DOC_TOKEN, line, line_number=file_reader.line_number + 1, throw=False):
                raise ParseError("Got unexpected `%s`  at line %d" %
                                 (DOC_TOKEN, file_reader.line_number + 1))
            elif consume_call_statement(FUNCTION_TOKEN, line, line_number=file_reader.line_number + 1, throw=False):
                self.children.append(NpeFunction(file_reader))
                assert len(self.children[-1].preamble) == 0
                self.children[-1].preamble = _preamble
                _preamble = ""
            else:
                _preamble += file_reader.readline()


def codegen_ast(ast, out_file, write_debug_prints=True):
    PRIVATE_ID_PREFIX = "_NPE_PY_BINDING_"
    PRIVATE_NAMESPACE = "npe::detail"
    STORAGE_ORDER_ENUM = "StorageOrder"
    ALIGNED_ENUM = "Alignment"
    TYPE_ID_ENUM = "TypeId"
    TYPE_CHAR_ENUM = "NumpyTypeChar"
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

    def cast_arg(var):
        if var.is_dense:
            return "static_cast<pybind11::array&>(%s)" % var.name
        elif var.is_sparse:
            return "static_cast<npe::sparse_array&>(%s)" % var.name
        else:
            raise AssertionError("This should never happen!")

    def type_name_var(var_name):
        return PRIVATE_ID_PREFIX + var_name + "_type_s"

    def storage_order_var(var_name):
        return PRIVATE_ID_PREFIX + var_name + "_so"

    def type_id_var(var_name):
        return PRIVATE_ID_PREFIX + var_name + "_t_id"

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

    def write_flags_getter(arg):
        storage_order_var_name = storage_order_var(arg.name)
        row_major = PRIVATE_NAMESPACE + "::RowMajor"
        col_major = PRIVATE_NAMESPACE + "::ColMajor"
        no_order = PRIVATE_NAMESPACE + "::NoOrder"
        out_str = "const " + PRIVATE_NAMESPACE + "::" + STORAGE_ORDER_ENUM + " " + storage_order_var_name + " = "
        out_str += "(" + cast_arg(arg) + ".flags() & NPY_ARRAY_F_CONTIGUOUS) ? " + col_major + " : "
        out_str += "(" + cast_arg(arg) + ".flags() & NPY_ARRAY_C_CONTIGUOUS ? " + row_major + " : " + no_order + ");\n"
        out_file.write(out_str)

    def write_type_id_getter(arg):
        out_str = "const int " + type_id_var(arg.name) + " = "
        type_name = type_name_var(arg.name)
        storage_order_name = storage_order_var(arg.name)
        is_sparse = PRIVATE_NAMESPACE + "::is_sparse<std::remove_reference<decltype(" + \
                    cast_arg(arg) + ")>::type>::value"
        out_str += PRIVATE_NAMESPACE + "::get_type_id(" + is_sparse + ", " + type_name + ", " + \
                   storage_order_name + ");\n"
        out_file.write(out_str)

    def write_function_switch_header(fun):
        out_file.write('"%s"' % fun.name)
        out_file.write(", [](")

        # Write the argument list
        i = 0
        for arg in fun.arguments:
            prefix = "npe::detail::maybe_none<" if arg.is_nullable else ""
            suffix = "> " if arg.is_nullable else " "

            if arg.is_sparse:
                out_file.write(prefix + "npe::sparse_array" + suffix)
                out_file.write(arg.name)
            elif arg.is_dense:
                out_file.write(prefix + "pybind11::array" + suffix)
                out_file.write(arg.name)
            else:
                # Non Numpy/Scipy arguments only have one type
                assert len(arg.types) == 1, "More than one type %s" % str(arg.types)
                var_type = arg.types[0]
                out_file.write(var_type + " ")
                out_file.write(arg.name)
            next_token = ", " if i < fun.num_arguments - 1 else ") {\n"
            out_file.write(next_token)
            i += 1

        if fun.num_arguments == 0:
            out_file.write(") {\n")

        out_file.write("#ifdef __NPE_REDIRECT_IO__\n")
        out_file.write('pybind11::scoped_ostream_redirect __npe_redirect_stdout__(std::cout, '
                       'pybind11::module::import("sys").attr("stdout"));\n')
        out_file.write('pybind11::scoped_ostream_redirect __npe_redirect_stderr__(std::cerr, '
                       'pybind11::module::import("sys").attr("stderr"));\n')
        out_file.write("#endif\n")

        if write_debug_prints:
            out_file.write('std::cout << "Invocation of %s" << std::endl;\n' % fun.name)

        # Declare variables used to determine the type at runtime
        for arg in fun.array_arguments:
            # A character representing the dtype of the argument
            out_file.write("const char %s = %s::transform_typechar(%s.dtype().type());\n" %
                           (type_name_var(arg.name), PRIVATE_NAMESPACE, cast_arg(arg)))

            # Check that the shape of the argument has length one or 2
            out_file.write("ssize_t %s_shape_0 = 0;\n" % arg.name)
            out_file.write("ssize_t %s_shape_1 = 0;\n" % arg.name)
            out_file.write("if (%s.ndim() == 1) {\n" % cast_arg(arg))
            out_file.write("%s_shape_0 = %s.shape()[0];\n" % (arg.name, cast_arg(arg)))
            out_file.write("%s_shape_1 = %s.shape()[0] == 0 ? 0 : 1;\n" % (arg.name, cast_arg(arg)))
            out_file.write("} else if (%s.ndim() == 2) {\n" % cast_arg(arg))
            out_file.write("%s_shape_0 = %s.shape()[0];\n" % (arg.name, cast_arg(arg)))
            out_file.write("%s_shape_1 = %s.shape()[1];\n" % (arg.name, cast_arg(arg)))
            out_file.write("} else if (%s.ndim() > 2) {\n" % cast_arg(arg))
            out_file.write("throw std::invalid_argument(\"Argument " + arg.name +
                           " has invalid number of dimensions. Must be 1 or 2.\");\n")
            out_file.write("}\n")

            # Declare variables representing the storage order and an integer ID representing the combination of
            # scalar type (dtype) and storage order of the argument
            write_flags_getter(arg)
            write_type_id_getter(arg)

            if write_debug_prints:
                out_file.write('std::cout << "- Argument: %s" << std::endl;\n' % arg.name)
                out_file.write('std::cout << "   - shape: (" << %s << ", " << %s << ")" << std::endl;\n' %
                               (str(arg.name + "_shape_0"), str(arg.name + "_shape_1")))

                storage_order_var_name = storage_order_var(arg.name)
                out_file.write('std::cout << "   - " << %s::storage_order_to_str(%s) << std::endl;\n' %
                               (PRIVATE_NAMESPACE, storage_order_var_name))

                type_name = type_name_var(arg.name)
                out_file.write('std::cout << "   - type char: " << %s << std::endl;\n' % type_name)
                out_file.write('std::cout << "   - type name: " << %s::type_to_str(%s) << std::endl;\n' %
                               (PRIVATE_NAMESPACE, type_name))

        if write_debug_prints:
            out_file.write('std::cout << "-------------------------------------------------------" << std::endl;\n')

        # Ensure the types in each group match
        first_non_nullables = []

        for grp in fun.argument_groups:
            # At least one argument must be non-nullable, find this argument and use it as a reprensentative to
            # compare the types of all arguments in a group to
            first_non_nullable = None
            for arg in grp.arguments:
                if not arg.is_nullable:
                    first_non_nullable = arg
                    break
            assert first_non_nullable is not None, "What in the actual fuck?"
            first_non_nullables.append(first_non_nullable)

            # TODO: Pretty error message highlighting exactly which arguments mismatch
            # Compare the type of the representative to every valid type in the group and raise an exception if there
            # is a mismatch
            out_str = "if ("
            for i in range(len(grp.types)):
                type_name = grp.types[i]
                out_str += type_name_var(first_non_nullable.name) + "!= " + \
                           PRIVATE_NAMESPACE + "::transform_typechar( " + type_char_for_numpy_type(type_name) + ")"
                next_token = " && " if i < len(grp.types) - 1 else ") {\n"
                out_str += next_token
            pretty_group_types = [NUMPY_ARRAY_TYPES_TO_CPP[gt][2] for gt in grp.types]
            out_str += 'std::string err_msg = std::string("Invalid scalar type (") + ' \
                       '%s::type_to_str(%s) + ", " + %s::storage_order_to_str(%s) + ' \
                       'std::string(") for argument \'%s\'. Expected one of %s.");\n' % \
                       (PRIVATE_NAMESPACE, type_name_var(first_non_nullable.name),
                        PRIVATE_NAMESPACE, storage_order_var(first_non_nullable.name),
                        first_non_nullable.name, pretty_group_types)
            out_str += 'throw std::invalid_argument(err_msg);\n'
            out_str += "}\n"
            out_file.write(out_str)

            # If there is only one argument to check, then don't generate any extra checks
            assert len(grp.arguments) >= 1
            if len(grp.arguments) == 1:
                continue

            # Now, check that the dtype and storage order of every argument in the group matches the type of the
            # representative.
            # If one of the arguments is a vector or a zero sized array, then automatically coalesce it to the right
            # storage order
            out_file.write("{\n")
            out_file.write("int group_matched_type_id = %s;\n" % type_id_var(first_non_nullable.name))
            out_file.write("bool found_non_1d = "
                           "(%s_shape_0 != 1 && %s_shape_1 != 1 && %s_shape_0 != 0 && %s_shape_1 != 0);\n" %
                           (first_non_nullable.name, first_non_nullable.name,
                            first_non_nullable.name, first_non_nullable.name))
            out_file.write('std::string match_to_name = "%s";\n' % first_non_nullable.name)
            out_file.write('%s::StorageOrder match_so = %s;\n' %
                           (PRIVATE_NAMESPACE, storage_order_var(first_non_nullable.name)))
            out_file.write("char group_type_s = %s;\n" % type_name_var(first_non_nullable.name))

            for arg in grp.arguments:
                exception_str1 = 'std::string err_msg = std::string("Invalid type (") + ' \
                                 '%s::type_to_str(%s) + ", " + %s::storage_order_to_str(%s) + ' \
                                 'std::string(") for argument \'%s\'. Expected it to match argument \'") + ' \
                                 'match_to_name + std::string("\' which is of type (") + ' \
                                 '%s::type_to_str(group_type_s) + ", " + %s::storage_order_to_str(match_so) + ' \
                                 'std::string(").");\n' \
                                 % (PRIVATE_NAMESPACE, type_name_var(arg.name),
                                    PRIVATE_NAMESPACE, storage_order_var(arg.name),
                                    arg.name, PRIVATE_NAMESPACE, PRIVATE_NAMESPACE) + \
                                 'throw std::invalid_argument(err_msg);\n'
                exception_str2 = 'std::string err_msg = std::string("Invalid type (") + ' \
                                 '%s::type_to_str(%s) + ", " + %s::storage_order_to_str(match_so) + ' \
                                 'std::string(") for argument \'%s\'. Expected it to match argument \'") + ' \
                                 'match_to_name + std::string("\' which is of type (") + ' \
                                 '%s::type_to_str(group_type_s) + ", " +  %s::storage_order_to_str(match_so) + ' \
                                 'std::string(").");\n' \
                                 % (PRIVATE_NAMESPACE, type_name_var(arg.name),
                                    PRIVATE_NAMESPACE, arg.name, PRIVATE_NAMESPACE, PRIVATE_NAMESPACE) + \
                                 'throw std::invalid_argument(err_msg);\n'
                out_str = "if (!" + arg.name + ".is_none) {\n" if arg.is_nullable else ""
                out_str += "if (" + str(arg.name + "_shape_0") + " != 1 && " + \
                           str(arg.name + "_shape_1") + " != 1 && " + \
                           str(arg.name + "_shape_0") + " != 0 && " + \
                           str(arg.name + "_shape_1") + " != 0) {\n"
                out_str += "if (!found_non_1d) {\n" + \
                           "group_matched_type_id = " + type_id_var(arg.name) + ";\n" + \
                           "found_non_1d = true;\n" + \
                           'match_to_name = "' + arg.name + '";\n' + \
                           "match_so = " + storage_order_var(arg.name) + ";\n" + \
                           "group_type_s = " + type_name_var(arg.name) + ";\n" + \
                           "\n}\n"
                out_str += "if (" + type_id_var(arg.name) + " != group_matched_type_id) {\n"
                out_str += exception_str1
                out_str += "}\n"
                out_str += "} else if (group_type_s != %s) {\n" % type_name_var(arg.name)
                out_str += exception_str2
                out_str += "}\n"
                out_str += "}\n" if arg.is_nullable else ""
                out_file.write(out_str)
            out_file.write("}\n")

        # Now check that each metagroup have matching dtypes
        for metagrp in fun.metagroups:
            rep_arg = first_non_nullables[metagrp.rep_group.id]
            for grp in metagrp.groups:
                if grp == rep_arg.group:
                    continue

                arg = first_non_nullables[grp.id]
                exception_str = 'std::string err_msg = std::string("Invalid dtype `") + ' \
                                '%s::type_to_str(%s) + ' 'std::string("` for argument \'%s\'. ' \
                                'Expected it to match argument ") + ' \
                                '"\'%s\'" + std::string(" which has dtype `") + ' \
                                '%s::type_to_str(%s) + ' \
                                'std::string("`.");\n' \
                                % (PRIVATE_NAMESPACE, type_name_var(arg.name), arg.name, rep_arg.name,
                                   PRIVATE_NAMESPACE, type_name_var(rep_arg.name)) + \
                                'throw std::invalid_argument(err_msg);\n'

                out_str = "if (!" + arg.name + ".is_none) {\n" if arg.is_nullable else ""
                out_str += "if(" + type_name_var(arg.name) + " != " + type_name_var(rep_arg.name) + ") {\n"
                out_str += exception_str
                out_str += "}\n"
                out_str += "}\n" if arg.is_nullable else ""
                out_file.write(out_str)

    def write_function_switch_body(fun):
        # At this stage in execution, we're guaranteed that every argument in a group has exactly the same type
        # and that any arguments in the same metagroup have exactly the same dtype
        expanded_type_groups = [itertools.product(group.types, STORAGE_ORDER_SUFFIXES) for group in fun.argument_groups]
        group_combos = itertools.product(*expanded_type_groups)
        branch_count = 0

        if fun.has_array_arguments:
            for combo in group_combos:
                if_or_elseif = "if " if branch_count == 0 else " else if "
                if platform.system() == "Windows":  # Windows has a branch limit so we don't use else if
                    if_or_elseif = "if"

                out_str = if_or_elseif + "("
                skip = False
                for group_id in range(len(combo)):
                    # Sparse types only have column (csc) and row (csr) matrix types,
                    #  so don't output a branch for unaligned
                    if is_sparse_type(combo[group_id][0]) and combo[group_id][1] == STORAGE_ORDER_SUFFIX_XM:
                        skip = True
                        break
                    repr_var = fun.argument_groups[group_id].arguments[0]
                    typename = combo[group_id][0] + combo[group_id][1]
                    out_str += type_id_var(repr_var.name) + " == " + PRIVATE_NAMESPACE + "::transform_typeid(" + \
                               PRIVATE_NAMESPACE + "::" + TYPE_ID_ENUM + "::" + typename + ")"
                    next_token = " && " if group_id < len(combo) - 1 else ")"
                    out_str += next_token

                def dtype_from_type(type_):
                    return type_.replace("dense_", "").replace("sparse_", "")

                for metagrp in fun.metagroups:
                    rep_dtype = dtype_from_type(combo[metagrp.groups[0].id][0])
                    for grp in metagrp.groups:
                        grp_dtype = dtype_from_type(combo[grp.id][0])
                        if grp_dtype != rep_dtype:
                            skip = True
                            break
                    else:
                        continue
                    break

                if skip:
                    continue

                out_str += " {\n"
                out_file.write(out_str)
                write_switch_branch(fun, combo)
                out_file.write("}")
                branch_count += 1

            throw_err = 'throw std::invalid_argument("This should never happen but clearly it did. ' \
                        'File a github issue at https://github.com/fwilliams/numpyeigen");\n'
            if platform.system() != "Windows":  # Windows has a branch limit so we don't do else if
                out_file.write(" else {\n")
                out_file.write(throw_err)
                out_file.write("}\n")
            else:
                out_file.write(throw_err)
        else:
            group_combos = list(group_combos)
            assert len(group_combos) == 1, "This should never happen but clearly it did. " \
                                           "File a github issue at https://github.com/fwilliams/numpyeigen"
            for _ in group_combos:
                out_file.write("{\n")
                out_file.write(fun.source_code + "\n")
                out_file.write("}\n")
        out_file.write("\n")
        out_file.write("}")

    def write_switch_branch(fun, combo):
        out_file.write("{\n")
        for group_id in range(len(combo)):
            type_prefix = combo[group_id][0]
            type_suffix = combo[group_id][1]
            for arg in fun.argument_groups[group_id].arguments:
                cpp_type = NUMPY_ARRAY_TYPES_TO_CPP[type_prefix][0]
                storage_order_enum = storage_order_for_suffix(type_suffix)
                aligned_enum = aligned_enum_for_suffix(type_suffix)

                out_file.write("typedef " + cpp_type + " Scalar_" + arg.name + ";\n")
                if is_sparse_type(combo[group_id][0]):
                    eigen_type = "Eigen::SparseMatrix<" + cpp_type + ", " + \
                                 storage_order_enum + ", int>"
                    out_file.write("typedef " + eigen_type + " Matrix_%s" % arg.name + ";\n")
                    out_file.write("#if EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION <= 2\n")
                    out_file.write("typedef Eigen::MappedSparseMatrix<" + cpp_type + ", " +
                                   storage_order_enum + ", int> Map_" + arg.name + ";\n")
                    out_file.write("#elif (EIGEN_WORLD_VERSION == 3 && "
                                   "EIGEN_MAJOR_VERSION > 2) || (EIGEN_WORLD_VERSION > 3)\n")
                    out_file.write("typedef Eigen::Map<Matrix_" + arg.name + "> Map_" + arg.name + ";\n")
                    out_file.write("#endif\n")

                else:
                    eigen_type = "Eigen::Matrix<" + cpp_type + ", " + "Eigen::Dynamic, " + "Eigen::Dynamic, " + \
                                 storage_order_enum + ">"
                    out_file.write("typedef " + eigen_type + " Matrix_%s" % arg.name + ";\n")
                    if type_suffix == STORAGE_ORDER_SUFFIX_XM:
                        out_file.write("Eigen::Index " + arg.name + "_inner_stride = 0;\n")
                        out_file.write("Eigen::Index " + arg.name + "_outer_stride = 0;\n")
                        out_file.write("if (" + arg.name + ".ndim() == 1) {\n")
                        out_file.write(arg.name + "_outer_stride = " + arg.name + ".strides(0) / sizeof(" +
                                       cpp_type + ");\n")
                        out_file.write("} else if (" + arg.name + ".ndim() == 2) {\n")
                        out_file.write(arg.name + "_outer_stride = " + arg.name + ".strides(1) / sizeof(" +
                                       cpp_type + ");\n")
                        out_file.write(arg.name + "_inner_stride = " + arg.name + ".strides(0) / sizeof(" +
                                       cpp_type + ");\n")
                        out_file.write("}")
                        # out_file.write('std::cout << "' + arg.name + ' strides = " << ' + arg.name +
                        #                '_outer_stride << ", " << ' + arg.name + '_inner_stride << std::endl;')
                        out_file.write("typedef Eigen::Map<" + eigen_type + ", " +
                                       aligned_enum + ", Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> Map_" +
                                       arg.name + ";\n")
                    else:
                        out_file.write("typedef Eigen::Map<" + eigen_type + ", " +
                                       aligned_enum + "> Map_" + arg.name + ";\n")

        call_str = "return callit_" + fun.name
        template_str = "<"
        for arg in fun.arguments:
            if arg.is_numpy_type:
                template_str += "Map_" + arg.name + ", Matrix_" + arg.name + ", Scalar_" + arg.name + ","

        template_str = template_str[:-1] + ">("

        call_str = call_str + template_str if fun.has_array_arguments else call_str + "("

        for arg in fun.arguments:
            if arg.is_numpy_type:
                stride_str = ""
                assert arg.group is not None
                arg_suffix = combo[arg.group.id][1]
                if arg_suffix == STORAGE_ORDER_SUFFIX_XM:
                    stride_str = ", Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(" + \
                                 arg.name + "_outer_stride, " + arg.name + "_inner_stride)"

                map_str = ""
                if arg.is_nullable:
                    map_str = arg.name + ".is_none ? Map_" + arg.name
                    if arg.is_sparse:
                        map_str += "(0, 0, 0, nullptr, nullptr, nullptr) : "
                    else:
                        map_str += "(nullptr, 0, 0" + stride_str + ") : "

                if not arg.is_sparse:
                    map_str += "Map_" + arg.name + "((Scalar_" + arg.name + "*) " + cast_arg(arg) + ".data(), " + \
                               arg.name + "_shape_0, " + arg.name + "_shape_1" + stride_str + "),"
                else:
                    map_str += arg.name + ".as_eigen<Matrix_" + arg.name + ">(),"

                call_str += map_str

            else:
                call_str += arg.name + ","

        call_str = call_str[:-1] + ");\n"
        out_file.write(call_str)
        # out_file.write(binding_source_code + "\n")
        out_file.write("}\n")

    def write_function_definition(fun):
        template_str = "template <"
        for arg in fun.arguments:
            if arg.is_numpy_type:
                template_str += "typename " + MAP_TYPE_PREFIX + arg.name + ","
                template_str += "typename " + MATRIX_TYPE_PREFIX + arg.name + ","
                template_str += "typename " + SCALAR_TYPE_PREFIX + arg.name + ","
        template_str = template_str[:-1] + ">\n"
        if fun.has_array_arguments:
            out_file.write(template_str)
        out_file.write("static auto callit_%s(" % fun.name)

        argument_str = ""
        for arg in fun.arguments:
            if arg.is_numpy_type:
                argument_str += "%s%s %s," % (MAP_TYPE_PREFIX, arg.name, arg.name)
            else:
                argument_str += arg.types[0] + " " + arg.name + ","
        argument_str = argument_str[:-1] + ") {\n"
        out_file.write(argument_str)
        out_file.write(fun.source_code)
        out_file.write("}\n")

    def write_function_switch_end(fun):
        if len(fun.docstring) > 0:
            out_file.write(", " + fun.docstring)

        arg_list = ""
        for arg in fun.arguments:
            arg_list += ", pybind11::arg(\"" + arg.name + "\")"

            if arg.is_nullable:
                arg_list += "=" + 'pybind11::none()'
            elif arg.default_value is not None:
                arg_list += "=" + arg.default_value

        out_file.write(arg_list)
        out_file.write(");\n")

    FOR_REAL_DEFINE = "__NPE_FOR_REAL__"
    out_file.write("#define " + FOR_REAL_DEFINE + "\n")
    out_file.write("#include <npe.h>\n")

    for child in ast.children:
        out_file.write(child.preamble + "\n")
        if type(child) == NpeFunction:
            write_function_definition(child)
        else:
            raise RuntimeError("What in the actual fuck?")

    func_name = "pybind_output_fun_" + os.path.basename(ast.input_file_name).replace(".", "_")
    out_file.write("void %s(pybind11::module& m) {\n" % func_name)

    for child in ast.children:
        out_file.write('m.def(')
        if type(child) == NpeFunction:
            write_function_switch_header(child)
            write_function_switch_body(child)
            write_function_switch_end(child)
        else:
            raise RuntimeError("What in the actual fuck?")

    out_file.write("}\n")
    out_file.write("\n")


def main():
    global cpp_command
    global cpp_path
    global verbosity_level

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("file", type=str)
    arg_parser.add_argument("cpp_cmd", type=str)
    arg_parser.add_argument("-o", "--output", type=str, default="a.out")
    arg_parser.add_argument("-v", "--verbosity-level", type=int, default=LOG_INFO,
                            help="How verbose is the output. < 0 = silent, "
                                 "0 = only errors, 1 = normal, 2 = verbose, > 3 = debug")
    arg_parser.add_argument('--debug-trace', action='store_true',
                            help='Print traces containing type information of variables passed into bound functions')
    arg_parser.add_argument('--c-preprocessor-args', help='Input String', nargs='*', type=str)

    args = arg_parser.parse_args()

    cpp_path = args.cpp_cmd

    cpp_command = []
    for tmp in args.c_preprocessor_args:
        for t in tmp.split(" "):
            cpp_command.append(t)

    verbosity_level = args.verbosity_level

    try:
        with NpeFileReader(args.file) as infile:
            ast = NpeAST(infile)
        with open(args.output, 'w+') as outfile:
            codegen_ast(ast, outfile, write_debug_prints=args.debug_trace)
    except SemanticError as e:
        # TODO: Pretty printer
        log(LOG_ERROR, TermColors.FAIL + TermColors.BOLD + "NumpyEigen Semantic Error: " +
            TermColors.ENDC + TermColors.ENDC + str(e), file=sys.stderr)
        sys.exit(1)
    except ParseError as e:
        # TODO: Pretty printer
        log(LOG_ERROR, TermColors.FAIL + TermColors.BOLD + "NumpyEigen Syntax Error: " +
            TermColors.ENDC + TermColors.ENDC + str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
