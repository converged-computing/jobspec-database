import numpy as np
import quantum_query_optimizer as qqo

def get_boolean_labels(p):
    labels = list(range(p)) + [np.inf]
    for row_index in range(p):
        for col_index in range(p):
            labels += [(row_index, col_index)]
    return labels

def points_on_line_with_slope(p, slope, y_intercept):
    ''' Returns a list of points on the line with the given slope and y-intercept. '''
    current = (0, y_intercept) # Start from bottom row
    points_on_line = [slope, current] # Include slope and first point
    for _ in range(p-1):
        # Move to next point on line
        # One to the right, slope up
        current = ((current[0] + slope) % p, (current[1]+1) % p)
        points_on_line += [current]
    return points_on_line

def points_on_vertical_line(p, x_intercept):
    ''' Returns a list of points on the vertical line with the given x-intercept. '''
    points_on_line = [np.inf]
    for y_intercept in range(p):
        points_on_line += [(x_intercept, y_intercept)]
    return points_on_line

def points_on_horizontal_line(p, y_intercept):
    ''' Returns a list of points on the horizontal line with the given y-intercept. '''
    points_on_line = [0]
    for x_intercept in range(p):
        points_on_line += [(x_intercept, y_intercept)]
    return points_on_line

def check_all_present(points_on_line, instance):
    ''' Returns True if all points on the line are present in the instance. '''
    #print('Checking points_on_line=', points_on_line)
    is_present = [instance[point] == '1' for point in points_on_line]
    #print(is_present)
    return all(is_present)

def is_line_present(p, instance):
    # Check horizontal lines
    for y_intercept in range(p):
        points_on_line = points_on_horizontal_line(p, y_intercept)
        if check_all_present(points_on_line, instance):
            return '1'
    # Check slopes between 1 and p-1
    for slope in range(1, p):
        for y_intercept in range(p):
            points_on_line = points_on_line_with_slope(p, slope, y_intercept)
            if check_all_present(points_on_line, instance):
                return '1'
    # Check vertical lines
    for x_intercept in range(p):
        points_on_line = points_on_vertical_line(p, x_intercept)
        if check_all_present(points_on_line, instance):
            return '1'
    # Check line at infinity
    points_on_line = [np.inf]
    for point in range(p):
        points_on_line += [point]     
    if check_all_present(points_on_line, instance):
        return '1'
    return '0'

def build_single_hard(p, points_on_line, correspondance):
    ''' Returns a list of p+1 instances, one positive and p negatives. '''
    num_points = p**2 + p + 1
    x = '0' * num_points
    for point in points_on_line:
        index = correspondance[point]
        x = x[:index] + '1' + x[index+1:]
    xs = [x]
    #print(x)
    # Perturb each point on the line for a negative instance
    for point in points_on_line:
        index = correspondance[point]
        new_x = x[:index] + '0' + x[index+1:]
        xs += [new_x]
    return xs

def build_hard_instances(p, correspondance):
    xs = []
    # Check horizontal lines
    for y_intercept in range(p):
        points_on_line = points_on_horizontal_line(p, y_intercept)
        xs += build_single_hard(p, points_on_line, correspondance)
    # Check slopes between 1 and p-1
    for slope in range(1, p):
        for y_intercept in range(p):
            points_on_line = points_on_line_with_slope(p, slope, y_intercept)
            xs += build_single_hard(p, points_on_line, correspondance)
    # Check vertical lines
    for x_intercept in range(p):
        points_on_line = points_on_vertical_line(p, x_intercept)
        xs += build_single_hard(p, points_on_line, correspondance)
    # Check line at infinity
    points_on_line = [np.inf]
    for point in range(p):
        points_on_line += [point]     
    xs += build_single_hard(p, points_on_line, correspondance)
    return xs

def build_hard_input_and_output(p):
    labels = get_boolean_labels(p)
    n = len(labels)
    print('Number of bits:', n)
    correspondance = dict(zip(labels, range(n)))

    xs = build_hard_instances(p, correspondance)
    
    # Negate each bit
    negated_xs = []
    for x in xs:
        negated_x = ''.join(['1' if x[index] == '0' else '0' for index in range(n)])
        negated_xs += [negated_x]

    D = xs + negated_xs
    print('Number of inputs/outputs:', len(D))
    E = [is_line_present(p, dict(zip(labels, x))) for x in D]
    return D, E

def build_all_input_and_output(p):
    labels = get_boolean_labels(p)
    n = len(labels)
    print('Number of bits:', n)
    D = qqo.getDAll(n)
    print('Number of inputs/outputs:', len(D))
    E = [is_line_present(p, dict(zip(labels, x))) for x in D]
    return D, E

D, E = build_all_input_and_output(p=3)
#D, E = build_hard_input_and_output(p=3)
solutions = qqo.runSDP(D=D, E=E, verbose=True)