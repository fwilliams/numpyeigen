import argparse


NUMPY_ARRAY_TYPES = ['mat_f32', 'mat_f64', 'mat_i8', 'mat_i16', 'mat_i32', 'mat_i64',
                     'mat_u8', 'mat_u16', 'mat_u32', 'mat_u64']
MATCHES_TOKEN = "matches"
INPUT_TOKEN = "igl_input"
OUTPUT_TOKEN = "igl_output"
BEGIN_CODE_TOKEN = "igl_begin_code"
END_CODE_TOKEN = "igl_end_code"
INCLUDE_TOKEN = "#include"
BINDING_INIT_TOKEN = "igl_binding"

bound_function_name = ""  # The name of the function we are binding
input_type_groups = []  # Set of allowed types for each group of variables
input_varname_to_group = {}  # Dictionary mapping input variable names to type groups
group_to_input_varname = {}  # Dictionary mapping type groups to input variable names TODO: Populate
input_variables = {}  # Dictionary mapping variable names to types
output_variables = {}  # Dictionary mapping output variables to types or input variables whose types they match
binding_source_code = ""  # The source code of the binding


class ParseError(Exception):
    pass


class SemanticError(Exception):
    pass


class VariableMetadata(object):
    def __init__(self, name, is_matches, name_or_type, line_number):
        self.name = name
        self.is_matches = is_matches
        self.name_or_type = name_or_type
        self.line_number = line_number

    def __repr__(self):
        return str(self.__dict__)


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
    idx = line.find('"', 1)
    str_token = line[1:idx]
    return str_token, line[idx+1:]


def parse_eol_token(line, line_number):
    if len(line.strip()) != 0:
        # TODO: Pretty error message
        raise ParseError("Expected end-of-line after ')' token on line %d" % line_number)
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


def parse_input_statement(line, line_number):
    global NUMPY_ARRAY_TYPES, MATCHES_TOKEN, INPUT_TOKEN
    global input_type_groups, input_varname_to_group, group_to_input_varname

    line = parse_token(line.strip(), INPUT_TOKEN, line_number=line_number, case_sensitive=False)
    line = parse_token(line.strip(), '(', line_number=line_number)

    var_name, line = parse_string_token(line.strip(), line_number=line_number)
    validate_identifier_name(var_name)
    line = parse_token(line.strip(), ',', line_number=line_number)

    var_types = []

    while True:
        type_str, line = parse_string_token(line.strip(), line_number=line_number)
        var_types.append(type_str)
        token, line = parse_one_of_tokens(line.strip(), [')', ','], line_number=line_number)
        if token == ')':
            if line.strip() != "":
                # TODO: Pretty error message
                raise ParseError("Expected end-of-line after ')' token on line %d" % line_number)
            break

    is_matches = False

    if len(var_types) == 0:
        # TODO: Pretty error message
        raise ParseError('%s("%s") got no type arguments' % (INPUT_TOKEN, var_name))
    elif len(var_types) > 1 or (len(var_types) == 1 and is_numpy_type(var_types[0])):
        # If there are more than one type, then we're binding a numpy array. Check that the types are valid.
        for type_str in var_types:
            if not is_numpy_type(type_str):
                # TODO: Pretty error message
                raise ParseError("Got invalid type, `%s` in %s() at line %d. "
                                 "If multiple types are specified, "
                                 "they must be one of %s" % (type_str, INPUT_TOKEN, line_number, NUMPY_ARRAY_TYPES))

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
            group_to_input_varname[group_id] = var_name
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

    input_variables[var_name] = VariableMetadata(name=var_name,
                                                 is_matches=is_matches,
                                                 name_or_type=var_types,
                                                 line_number=line_number)

    return var_name, var_types


def parse_output_statement(line, line_number):
    global MATCHES_TOKEN, OUTPUT_TOKEN
    global input_varname_to_group, input_variables, output_variables

    # An output token is either a fixed type or a matches()
    line = parse_token(line.strip(), OUTPUT_TOKEN, line_number=line_number, case_sensitive=False)
    line = parse_token(line.strip(), '(', line_number=line_number)

    var_name, line = parse_string_token(line.strip(), line_number=line_number)
    validate_identifier_name(var_name)
    line = parse_token(line.strip(), ',', line_number=line_number)
    var_type, line = parse_string_token(line.strip(), line_number=line_number)

    if var_type.startswith(MATCHES_TOKEN):
        matches_name = parse_matches_statement(var_type.strip(), line_number=line_number)
        output_variables[var_name] = VariableMetadata(name=var_name,
                                                      is_matches=True,
                                                      name_or_type=matches_name,
                                                      line_number=line_number)
    else:
        output_variables[var_name] = VariableMetadata(name=var_name,
                                                      is_matches=False,
                                                      name_or_type=var_type,
                                                      line_number=line_number)

    line = parse_token(line.strip(), ')', line_number=line_number)

    parse_eol_token(line, line_number=line_number)

    return var_name, var_type


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

    line = parse_token(line.strip(), BINDING_INIT_TOKEN, line_number=line_number, case_sensitive=False)
    line = parse_token(line.strip(), '(', line_number=line_number)
    binding_name, line = parse_string_token(line.strip(), line_number=line_number)
    validate_identifier_name(binding_name)
    line = parse_token(line.strip(), ')', line_number=line_number)
    parse_eol_token(line.strip(), line_number=line_number)
    return binding_name


def frontend_pass(lines):
    global INPUT_TOKEN, OUTPUT_TOKEN, BEGIN_CODE_TOKEN, END_CODE_TOKEN, INCLUDE_TOKEN, BINDING_INIT_TOKEN
    global binding_source_code, bound_function_name

    binding_start_line_number = -1

    for line_number in range(len(lines)):
        if len(lines[line_number].strip()) == 0:
            continue
        elif lines[line_number].strip().lower().startswith(INCLUDE_TOKEN):
            # You can #include things to make your IDE work but the includes get ignored
            continue
        elif lines[line_number].strip().lower().startswith(BINDING_INIT_TOKEN):
            bound_function_name = parse_binding_init_statement(lines[line_number], line_number=line_number)
            binding_start_line_number = line_number + 1
            break
        else:
            raise ParseError("Unexpected tokens at line %d: %s" % (line_number, lines[line_number]))

    if binding_start_line_number < 0:
        raise ParseError("Invalid binding file. Must begin with %s(<function_name>)." % BINDING_INIT_TOKEN)

    print("Function: %s" % bound_function_name)

    code_start_line_number = -1

    for line_number in range(binding_start_line_number, len(lines)):
        if lines[line_number].strip().lower().startswith(INPUT_TOKEN):
            var_name, var_types = parse_input_statement(lines[line_number], line_number=line_number)
            print("Input %s: %s" % (var_name, var_types))
        elif lines[line_number].strip().lower().startswith(OUTPUT_TOKEN):
            var_name, var_type = parse_output_statement(lines[line_number], line_number=line_number)
            print("Output %s: %s" % (var_name, var_type))
        elif lines[line_number].strip().lower().startswith(BEGIN_CODE_TOKEN):
            parse_begin_code_statement(lines[line_number], line_number=line_number)
            code_start_line_number = line_number + 1
            break
        elif len(lines[line_number].strip()) == 0:
            # Ignore newlines and whitespace
            continue
        else:
            raise ParseError("Unexpected tokens at line %d: %s" % (line_number, lines[line_number]))

    if code_start_line_number < 0:
        raise ParseError("Invalid binding file. Must does not contain a %s() statement." % BEGIN_CODE_TOKEN)

    reached_end_token = False
    for line_number in range(code_start_line_number, len(lines)):
        if lines[line_number].lower().startswith(END_CODE_TOKEN):
            parse_end_code_statement(lines[line_number], line_number=line_number)
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
    global input_type_groups, input_variables, output_variables

    for var_name in input_variables.keys():
        var_meta: VariableMetadata = input_variables[var_name]
        if var_meta.is_matches:
            group_idx = input_varname_to_group[var_name]
            matches_name = var_meta.name_or_type[0]
            if len(input_type_groups[group_idx]) == 0:
                raise SemanticError("Input Variable %s (line %d) was declared with type %s but was "
                                    "unmatched with a numpy type." % (var_name, var_meta.line_number, matches_name))

    for var_name in output_variables.keys():
        var_meta: VariableMetadata = output_variables[var_name]
        if var_meta.is_matches:
            matches_name = var_meta.name_or_type
            if matches_name not in input_varname_to_group:
                raise SemanticError("Output variable %s type, %s(%s) must match a valid input variable at line %d" %
                                    (var_name, MATCHES_TOKEN, matches_name, var_meta.line_number))


def backend_pass():
    # TODO: Codegen the backend
    # For each vartype group compute product between type and the three packing orders
    # Generate all type combos
    # For each combo generate a branch in the if statement
    pass


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("file", type=str)
    arg_parser.add_argument("-o", type=str, default="a.out")

    args = arg_parser.parse_args()

    with open(args.file, 'r') as f:
        line_list = f.readlines()

    frontend_pass(line_list)
    validate_frontend_output()
    backend_pass()

    print(input_type_groups)
    print(input_varname_to_group)
    print(group_to_input_varname)
