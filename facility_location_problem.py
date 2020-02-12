from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('mip_program',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

infinity = solver.infinity()

c = {(1, 1): 4, (1, 2): 6, (1, 3): 9,  # unit cost of transportation, format: (destination, source)
     (2, 1): 5, (2, 2): 4, (2, 3): 7,
     (3, 1): 6, (3, 2): 3, (3, 3): 4,
     (4, 1): 8, (4, 2): 5, (4, 3): 3,
     (5, 1): 10, (5, 2): 8, (5, 3): 4,
     }

d = (80, 270, 250, 160, 180)  # amount required
m = (500, 500, 500)  # each source's largest  transportation capacity
f = (1000, 1000, 1000)  # each source's initialization cost

x = {}  # units of goods transported, format: (destination, source)
for k, v in c.items():
    string = 'x' + str(k[0])+str(k[1])
    x[k] = solver.NumVar(0, infinity, string)

y1 = solver.IntVar(0, 1, 'y1')
y2 = solver.IntVar(0, 1, 'y2')
y3 = solver.IntVar(0, 1, 'y3')

y_list = [y1, y2, y3]  # 01 variable, indicating whether corresponding source is initialized

for col in range(1, 6):  # each destination should at least receive d[col -1] goods required
    temp = None
    for row in range(1, 4):
        if temp is None:
            temp = x[(col, row)]
        else:
            temp += x[(col, row)]
    solver.Add(temp == d[col - 1])

for row in range(1, 4):  # each goods transported from each source should not exceed the transportation capacity
    temp = None
    for col in range(1, 6):
        if temp is None:
            temp = x[(col, row)]
        else:
            temp += x[(col, row)]
    solver.Add(temp <= m[row - 1])

for row in range(1, 4):  # each source's transportation amount also influenced by corresponding y
    temp = None
    for col in range(1, 6):
        if temp is None:
            temp = x[(col, row)]
        else:
            temp += x[(col, row)]
    solver.Add(temp <= y_list[row - 1] * m[row - 1])

min1 = None
for row in range(1, 4):  # minimize cost of initialize sources
    if min1 is None:
        min1 = y_list[row - 1] * f[row - 1]
    else:
        min1 += y_list[row - 1] * f[row - 1]

min2 = None  # minimize cost of transportation
for row in range(1, 4):
    for col in range(1, 6):
        if min2 is None:
            min2 = x[(col, row)] * c[(col, row)]
        else:
            min2 += x[(col, row)] * c[(col, row)]

solver.Minimize(min1 + min2)
status = solver.Solve()
print(status)
for y in y_list:
    print(y.solution_value())
