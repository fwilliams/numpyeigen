import time
import numpy as np
import pyigl_proto
import matplotlib.pyplot as plt

all_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
             np.float32, np.float64, np.complex64, np.complex128, np.complex256]
v_types = [np.float32, np.float64]
f_types = [np.int32, np.int64, np.uint32, np.uint64]
f_types2 = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]

base_times = []
for i in range(10000):
    v1 = np.random.rand(1000, 2)
    v1 = v1.astype(np.random.choice(all_types))
    if np.random.rand() > 0.5:
        v1 = np.asfortranarray(v1)
    tbeg = time.time()
    pyigl_proto.return_an_int(v1)
    tend = time.time()
    base_times.append(tend-tbeg)

print("mean-base_times:", np.mean(base_times))
print("std-base_times:", np.std(base_times))
print("max-base_times:", np.max(base_times))
print("min-base_times:", np.min(base_times))


lookup_times = []
for i in range(10000):
    v1 = np.random.rand(1000, 2)
    v1 = v1.astype(np.random.choice(all_types))
    if np.random.rand() > 0.5:
        v1 = np.asfortranarray(v1)
    tbeg = time.time()
    pyigl_proto.type_lookup(v1)
    tend = time.time()
    lookup_times.append(tend-tbeg)
print()
print("mean-lookup_times:", np.mean(lookup_times))
print("std-lookup_times:", np.std(lookup_times))
print("max-lookup_times:", np.max(lookup_times))
print("min-lookup_times:", np.min(lookup_times))

base_cot_times = []
for i in range(10000):
    v1 = np.random.rand(1000, 3).astype(np.float32)
    v1 = v1.astype(np.random.choice(all_types))
    f = np.zeros([500, 3], dtype=np.int32)
    tbeg = time.time()
    pyigl_proto.cotmatrix(v1, f)
    tend = time.time()
    base_cot_times.append(tend-tbeg)
    
print()
print("mean-base_cot_times:", np.mean(base_cot_times))
print("std-base_cot_times:", np.std(base_cot_times))
print("max-base_cot_times:", np.max(base_cot_times))
print("min-base_cot_times:", np.min(base_cot_times))

fancy_cot_times = []
for i in range(100000):
    v1 = np.random.rand(1000, 3)
    v1 = v1.astype(np.random.choice(v_types))
    if np.random.rand() > 0.5:
        v1 = np.asfortranarray(v1)
    else:
        v1 = np.ascontiguousarray(v1)

    f = np.zeros([500, 3], dtype=np.random.choice(f_types))
    if np.random.rand() > 0.5:
        f = np.asfortranarray(f)
    else:
        f = np.ascontiguousarray(f)

    tbeg = time.time()
    pyigl_proto.cotmatrix_branch(v1, f)
    tend = time.time()
    fancy_cot_times.append(tend-tbeg)

print()
print("mean-fancy-cot-times:", np.mean(fancy_cot_times))
print("std-fancy-cot-times:", np.std(fancy_cot_times))
print("max-fancy-cot-times:", np.max(fancy_cot_times))
print("min-fancy-cot-times:", np.min(fancy_cot_times))

fancy_cot_times2 = []
for i in range(100000):
    v1 = np.random.rand(1000, 3)
    v1 = v1.astype(np.random.choice(v_types))
    if np.random.rand() > 0.5:
        v1 = np.asfortranarray(v1)
    else:
        v1 = np.ascontiguousarray(v1)

    f = np.zeros([500, 3], dtype=np.random.choice(f_types))
    if np.random.rand() > 0.5:
        f = np.asfortranarray(f)
    else:
        f = np.ascontiguousarray(f)

    tbeg = time.time()
    pyigl_proto.cotmatrix_branch2(v1, f)
    tend = time.time()
    fancy_cot_times2.append(tend-tbeg)

print()
print("mean-fancy-cot-times2:", np.mean(fancy_cot_times2))
print("std-fancy-cot-times2:", np.std(fancy_cot_times2))
print("max-fancy-cot-times2:", np.max(fancy_cot_times2))
print("min-fancy-cot-times2:", np.min(fancy_cot_times2))


fancy_cot_times3 = []
c_list = []
for i in range(100000):
    v1 = np.random.rand(1000, 3)
    v1 = v1.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v1 = np.asfortranarray(v1)
    elif 0.3333333333333 <= selector < 2*0.3333333333333:
        v1 = np.ascontiguousarray(v1)
    else:
        v1 = v1[0:1000:3, :]

    f = np.zeros([500, 3], dtype=np.random.choice(f_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        f = np.asfortranarray(f)
    elif 0.3333333333333 <= selector < 2*0.3333333333333:
        f = np.ascontiguousarray(f)
    else:
        f = f[0:500:3, :]

    tbeg = time.time()
    _, _, c = pyigl_proto.cotmatrix_branch3(v1, f)
    tend = time.time()
    fancy_cot_times3.append(tend-tbeg)
    c_list.append(c)

print()
print("mean-fancy-cot-times3:", np.mean(fancy_cot_times3))
print("std-fancy-cot-times3:", np.std(fancy_cot_times3))
print("max-fancy-cot-times3:", np.max(fancy_cot_times3))
print("min-fancy-cot-times3:", np.min(fancy_cot_times3))


fancy_cot_times4 = []
c_list2 = []
for i in range(100000):
    v1 = np.random.rand(1000, 3)
    v1 = v1.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v1 = np.asfortranarray(v1)
    elif 0.3333333333333 <= selector < 2*0.3333333333333:
        v1 = np.ascontiguousarray(v1)
    else:
        v1 = v1[0:1000:3, :]

    v2 = np.random.rand(1000, 3)
    v2 = v2.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v2 = np.asfortranarray(v2)
    elif 0.3333333333333 <= selector < 2*0.3333333333333:
        v2 = np.ascontiguousarray(v2)
    else:
        v2 = v2[0:1000:3, :]
        
    v3 = np.random.rand(1000, 3)
    v3 = v3.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v3 = np.asfortranarray(v3)
    elif 0.3333333333333 <= selector < 2*0.3333333333333:
        v3 = np.ascontiguousarray(v3)
    else:
        v3 = v3[0:1000:3, :]

    v4 = np.random.rand(1000, 3)
    v4 = v4.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v4 = np.asfortranarray(v4)
    elif 0.3333333333333 <= selector < 2*0.3333333333333:
        v4 = np.ascontiguousarray(v4)
    else:
        v4 = v4[0:1000:3, :]

    tbeg = time.time()
    _, _, _, _, c = pyigl_proto.cotmatrix_branch4(v1, v2, v3, v4)
    tend = time.time()
    fancy_cot_times4.append(tend-tbeg)
    c_list2.append(c)

print()
print("mean-fancy-cot-times4:", np.mean(fancy_cot_times4))
print("std-fancy-cot-times4:", np.std(fancy_cot_times4))
print("max-fancy-cot-times4:", np.max(fancy_cot_times4))
print("min-fancy-cot-times4:", np.min(fancy_cot_times4))

fancy_cot_times5 = []
c_list3 = []
for i in range(100000):
    v1 = np.random.rand(1000, 3)
    v1 = v1.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v1 = np.asfortranarray(v1)
    elif 0.3333333333333 <= selector < 2 * 0.3333333333333:
        v1 = np.ascontiguousarray(v1)
    else:
        v1 = v1[0:1000:3, :]

    v2 = np.random.rand(1000, 3)
    v2 = v2.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v2 = np.asfortranarray(v2)
    elif 0.3333333333333 <= selector < 2 * 0.3333333333333:
        v2 = np.ascontiguousarray(v2)
    else:
        v2 = v2[0:1000:3, :]

    v3 = np.random.rand(1000, 3)
    v3 = v3.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v3 = np.asfortranarray(v3)
    elif 0.3333333333333 <= selector < 2 * 0.3333333333333:
        v3 = np.ascontiguousarray(v3)
    else:
        v3 = v3[0:1000:3, :]

    v4 = np.random.rand(1000, 3)
    v4 = v4.astype(np.random.choice(v_types))
    selector = np.random.rand()
    if 0.0 <= selector < 0.3333333333333:
        v4 = np.asfortranarray(v4)
    elif 0.3333333333333 <= selector < 2 * 0.3333333333333:
        v4 = np.ascontiguousarray(v4)
    else:
        v4 = v4[0:1000:3, :]

    tbeg = time.time()
    _, _, _, _, c = pyigl_proto.cotmatrix_branch4(v1, v2, v3, v4)
    tend = time.time()
    fancy_cot_times5.append(tend - tbeg)
    c_list2.append(c)

print()
print("mean-fancy-cot-times5:", np.mean(fancy_cot_times5))
print("std-fancy-cot-times5:", np.std(fancy_cot_times5))
print("max-fancy-cot-times5:", np.max(fancy_cot_times5))
print("min-fancy-cot-times5:", np.min(fancy_cot_times5))

plt.hist(c_list2)
plt.show()
