from __future__ import print_function
from __future__ import division
from ...algorithms import sort
import numpy as np
import warnings

GREATEST_INT = 65535


class Container(object):
    def __init__(self):
        self.parent = []
        self.rank = []
        self.elements = []


def component_dict_num_pixel_map(C, shape=None):
    """Calculate total pixel for each component map"""
    the_new_dict = {}
    the_min_x = {}
    the_min_y = {}
    the_max_x = {}
    the_max_y = {}
    unique = set(C)
    C = np.array(C).reshape(shape)
    # Addin aspect ratio...
    for i in unique:
        the_new_dict[i] = 0
        the_min_x[i] = shape[1]
        the_min_y[i] = shape[0]
        the_max_x[i] = 0
        the_max_y[i] = 0

    for y in range(C.shape[0]):
        for x in range(C.shape[1]):
            i = C[y, x]
            the_new_dict[i] += 1
            the_min_x[i] = min(the_min_x[i], x)
            the_min_y[i] = min(the_min_y[i], y)
            the_max_x[i] = max(the_max_x[i], x)
            the_max_y[i] = max(the_max_y[i], y)

    return the_new_dict, the_min_x, the_max_x, the_min_y, the_max_y


class TreePixelNode:
    def __init__(self, level, i):
        self.parent = None
        self.children = []
        self.level = level
        self.index = i
        self.cumm_area = 0
        self.var = 0
        self.num_pixels_this_level = 0
        self.calculated_num_pixels = False
        self.calculated_area = False
        self.calculated_var = False
        self.min_x = 0
        self.min_y = 0
        self.max_y = 0
        self.max_x = 0
        self.is_mser = False

    def calculate_num_pixels_this_level(self, pixel_count_map,
                                        min_max_map):
        """ preorder walking to calculate number of pixels each level"""
        # Can confuse between C[self.index] & self.index since
        # this element is also the canonical element
        for child in self.children:
            child.parent = self
            child.calculate_num_pixels_this_level(pixel_count_map,
                                                  min_max_map)

        if len(self.children) > 0:
            self.min_x = min(min([child.min_x for child in self.children]),
                             min_max_map[0][self.index])
            self.max_x = max(max([child.max_x for child in self.children]),
                             min_max_map[1][self.index])
            self.min_y = min(min([child.min_y for child in self.children]),
                             min_max_map[2][self.index])
            self.max_y = max(max([child.max_y for child in self.children]),
                             min_max_map[3][self.index])
        else:
            self.min_x, self.max_x, self.min_y, self.max_y =\
                min_max_map[0][self.index], min_max_map[1][self.index],\
                min_max_map[2][self.index], min_max_map[3][self.index]
        self.aspect_ratio = (self.max_x - self.min_x+1)/(
            self.max_y - self.min_y + 1)
        self.num_pixels_this_level = pixel_count_map[self.index]

        self.calculated_num_pixels = True

    def calculate_area(self):
        """ postorder walking to calculate cummulated area"""
        for child in self.children:
            child.calculate_area()
            self.cumm_area += child.cumm_area
        self.cumm_area += self.num_pixels_this_level
        self.calculated_area = True

    def calculate_var(self, delta=2, pixel_count_map=None,
                      calling_on_child=False):
        """ calculate variance by passing down delta"""
        if not self.calculated_num_pixels:
            warnings.warn("TreePixelNode::Calculate Variance: \
                          Prestep calculate_num_pixels_this_level hasn\'t\
                          been called, Calling now...")
            if pixel_count_map is None:
                raise ValueError("No Component mapping available for\
                                 calculate_num_pixels_this_level")
            self.calculate_num_pixels_this_level(pixel_count_map)

        if not self.calculated_area:
            warnings.warn("TreePixelNode::Calculate Variance: \
                          Prestep calculate_area has not been called, Calling\
                          now...")
            self.calculate_area()

        sum_parent = 0
        the_current_child = self
        for i in range(delta):
            if the_current_child.parent is None:
                return GREATEST_INT
            sum_parent += the_current_child.parent.calculated_num_pixels
        self.var = sum_parent/self.cumm_area
        self.calculated_var = True
        if calling_on_child:
            for child in self.children:
                child.calculate_var(delta, calling_on_child=calling_on_child)

    def recursive_find_mser(self):
        if self.parent is not None and len(self.children > 0):
            if self.var < self.parent:
                pass


def make_set(Q, index):
    Q.parent.append(index)
    Q.elements.append(index)
    Q.rank.append(0)


def find(Q, x):
    """
    Input: x: index
    Return: index
    """
    x = Q.elements[x]
    if (Q.parent[x] != x):
        Q.parent[x] = find(Q, Q.parent[x])
    return Q.parent[x]


def link(Q, x, y):
    """
    Input: x,y : index
    """
    if Q.rank[x] > Q.rank[y]:
        x, y = y, x
    if Q.rank[x] == Q.rank[y]:
        Q.rank[y] += 1
    Q.parent[x] = y
    return y


def adj8(img_shape, i):
    """
    Input:
        img_shape: the shape of image
        i: index in the image (flattened)
    """
    the_list = []
    if (i % img_shape[1] != 0):
        the_list.extend([i-1])
    if ((i+1) % img_shape[1] != 0):
        the_list.extend([i+1])
    the_list.extend([i-img_shape[1], i+img_shape[1]])
    the_list = [int(x) for x in the_list if x >= 0 and
                x < (img_shape[0]*img_shape[1] - 1)]
    return the_list


def adj4(img_shape, i):
    """
    Input:
        img_shape: the shape of image
        i: index in the image (flattened)
    """
    the_list = []
    if (i % img_shape[1] != 0):
        the_list.extend([i-1, i-img_shape[1]-1, i+img_shape[1]-1])
    if ((i+1) % img_shape[1] != 0):
        the_list.extend([i+1, i-img_shape[1]+1, i+img_shape[1]+1])
    the_list.extend([i-img_shape[1], i+img_shape[1]])
    the_list = [int(x) for x in the_list if x >= 0 and
                x < (img_shape[0]*img_shape[1] - 1)]
    return the_list


def build_connected_component(img_shape, list_of_pixels, adjacent_pixels):
    """Connected component implementation
    Input:
        img_shape: shape of the input image
        list_of_pixels: the list of pixels to be added to connected component
        collection
        adjacent_pixels: a mapping that takes input as a tuple (height, width),
        and an index of the pixel and return indices of adjacent pixels
    Output:
        Q: a dictionary with key as element of list_of_pixels,
        the parent index of the node may be retrieved by calling
        find(Q, x)
    """
    Q = {}
    masking = np.zeros(np.prod(img_shape))
    for i in list_of_pixels:
        make_set(Q, i)
        masking[i] = 1
    for i in list_of_pixels:
        compp = find(Q, i)
        print("commpp index: ", compp)
        for q in adjacent_pixels(img_shape[:2], i):
            # This if can be optimized
            if masking[q] == 1:                         # q in list of pixels
                compq = find(Q, q)
                if compp != compq:
                    compp = link(Q, compq, compp)
    return Q


def make_node(level, i):
    return TreePixelNode(level, i)


def merge_nodes(nodes, Q2, node1_index, node2_index):
    node1 = Q2.elements[node1_index]
    node2 = Q2.elements[node2_index]
    tmp_node = link(Q2, node1, node2)
    if (tmp_node == node2):
        # Should consider C++ for faster implementation
        nodes[node2_index].children.extend(nodes[node1_index].children)
        nodes[node1_index] = 0  # No longer used
    else:
        nodes[node1_index].children.extend(nodes[node2_index].children)
        nodes[node2_index] = 0  # No longer used
    return tmp_node


def build_component_tree(img_gray, adjacent_pixels=adj8):
    """
    Input:
        adjacent_pixels: function with 2 argument: tuple(height, width),
                                 index of pixel in flattened image i
                         to return list of adjacent_pixels
        img_gray: input image
    Output:
        num_nodes: numper of nodes
        C: component mapping
        root: TreePixelNode
    """
    Q1 = Container()
    Q2 = Container()
    subtree_root = []
    nodes = []
    img_gray_flat = img_gray.flatten()
    indices = reversed(sort.counting_sort_indices(
        img_gray_flat, [[] for i in range(256)]))
    num_nodes = img_gray_flat.shape[0]
    for i in range(len(img_gray_flat)):
        make_set(Q1, i)
        make_set(Q2, i)
        nodes.append(make_node(img_gray_flat[i], i))
        subtree_root.append(i)

    for i in indices:
        curr_canonical_elt = find(Q1, i)
        curr_node = find(Q2, subtree_root[curr_canonical_elt])
        pixels = adj8(img_gray.shape, i)
        pixels = [pixel for pixel in pixels
                  if img_gray_flat[pixel] >= img_gray_flat[i]]
        for q in pixels:
            adj_canonical_elt = find(Q1, q)
            adj_node = find(Q2, subtree_root[adj_canonical_elt])
            if (curr_node != adj_node):
                if (nodes[curr_node].level == nodes[adj_node].level):
                    curr_node = merge_nodes(nodes, Q2, adj_node, curr_node)
                    num_nodes -= 1
                else:
                    nodes[curr_node].children.append(nodes[adj_node])
                curr_canonical_elt = link(Q1, adj_canonical_elt,
                                          curr_canonical_elt)
                subtree_root[curr_canonical_elt] = curr_node
    root = nodes[subtree_root[find(Q1, find(Q2, 0))]]
    C = []
    for i in range(len(img_gray_flat)):
        C.append(find(Q2, i))
    C = np.array(C)
    Q1 = None
    Q2 = None
    nodes[:] = []
    return num_nodes, root, C
