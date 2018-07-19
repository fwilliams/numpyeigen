import itertools

v_types = ['type_f32', 'type_f64']
f_types = ['type_i8', 'type_i16', 'type_i32', 'type_i64', 'type_u8', 'type_u16', 'type_u32', 'type_u64']
suffixes = ['_cm', '_rm', '_x']


def gen_bind(var_names, vartype_list):
    suffixes = ['_cm', '_rm', '_x']
    all_types = [itertools.product(vartypes, suffixes) for vartypes in vartype_list]

    # for x in all_types:
    #     print("Begin type list")
    #     for y in x:
    #         print(y)
    #     print()

    all_type_combos = itertools.product(*all_types)
    # for combo in all_type_combos:
    #     print(combo)

    count = 0

    for combo in all_type_combos:
        if_or_elseif = "if " if count == 0 else "} else if "
        clause = if_or_elseif + "("
        for i in range(len(var_names)):
            type_i = combo[i][0] + combo[i][1]
            clause += "tid%d == %s && " % (i, type_i)
        clause = clause[0:-4] + ") {"
        print(clause)

        code = ""
        for i in range(len(var_names)):
            suffix = combo[i][1]
            type_i = combo[i][0] + combo[i][1]
            var_name = var_names[i]
            data_var_name = var_name + "_data"
            shape_var_name = var_name + "_shape"
            eigen_var_name = var_name + "_eigen"
            stride_var_name = var_name + "_strides"
            if suffix != '_x':
                code_i = "    {0}::Scalar* {1} = ({0}::Scalar*)({2}.data(0));\n" \
                         "    Eigen::Map<{0}, Eigen::Aligned> {4}({1}, {3}[0], {3}[1]);\n".format(
                    type_i.capitalize(), data_var_name, var_name, shape_var_name, eigen_var_name)
                code_i = code_i
                code += code_i
            else:
                code_i =  "    {0}::Scalar* {1} = ({0}::Scalar*)({2}.data(0));\n" \
                          "    Eigen::Map<{0}, Eigen::Aligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> \n" \
                          "           {4}({1}, {3}[0], {3}[1],\n" \
                          "            Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>({5}[0], {5}[1]));\n".format(
                    type_i.capitalize(), data_var_name, var_name, shape_var_name, eigen_var_name, stride_var_name)
                code_i = code_i
                code += code_i

        code += "    return std::make_tuple<"

        for i in range(len(var_names)):
            code += "int, "
        code += "int>("
        for i in range(len(var_names)):
            var_name = var_names[i]
            eigen_var_name = var_name + "_eigen"
            code += eigen_var_name + ".rows(), "
        code += "%d);\n" % count
        count += 1

        print(code)
    print("} else {")
    errmsg =  "    cerr << \"Type not supported!\" << endl;\n"
    errmsg += "    cerr"
    for i in range(len(var_names)):
        errmsg += " << tid%d << \", \"" % i

    errmsg = errmsg[:-7] + " << endl;"
    print(errmsg)
    
    print("}")


f_types = ['type_i32', 'type_i64', 'type_u32', 'type_u64']
gen_bind(["v1", "v2", "v3", "v4", "v5"], [v_types, v_types, v_types, v_types, v_types])