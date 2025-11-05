#!/usr/bin/env python3
"""
Process llama_*.dot files to remove nodes with 'view(', 'reshape(', or 'permute(' in their labels.
When a node is removed, its parent nodes are connected directly to its child nodes.
Usage example: python3 scripts/process_dot_files.py llama_0.dot -o output_dot --suffix _processed [--max-layers 1]
"""

import networkx as nx
import argparse
import glob
import os
import re
from collections import deque


def should_remove_node(node_label):
    """
    Check if a node should be removed based on its label.
    Returns True if the label contains 'view(', 'reshape(', or 'permute('.
    """
    keywords = ['view(', 'reshape(', 'permute(']
    label_lower = node_label.lower()
    return any(keyword in label_lower for keyword in keywords)


def parse_dot_file(filepath):
    """
    Parse a DOT file and return a NetworkX DiGraph.
    """
    try:
        # Try using pydot first
        try:
            graph = nx.nx_pydot.read_dot(filepath)
            return graph
        except:
            pass
        
        # Try pygraphviz
        try:
            import pygraphviz
            graph = nx.nx_agraph.read_dot(filepath)
            return graph
        except ImportError:
            pass
        
        # Fallback to reading manually
        graph = read_dot_manually(filepath)
        return graph
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def read_dot_manually(filepath):
    """
    Manually parse a DOT file and create a NetworkX DiGraph.
    """
    graph = nx.DiGraph()
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Process line by line
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and graph definition lines
        if not line or line.startswith('digraph') or line == '}' or line.startswith('newrank') or line.startswith('rankdir'):
            continue
        
        # Check if it's a node definition (contains [ and label=)
        if '[' in line and 'label=' in line:
            # Extract node ID (first quoted string)
            node_match = re.match(r'"([^"]+)"\s*\[', line)
            if node_match:
                node_id = node_match.group(1)
                
                # Extract label (everything between label=" and the next ")
                # Handle complex labels with | and other characters
                label_match = re.search(r'label="([^"]*(?:\|[^"]*)*)"', line)
                if label_match:
                    label = label_match.group(1)
                    graph.add_node(node_id, label=label)
        
        # Check if it's an edge definition (contains ->)
        elif '->' in line:
            edge_match = re.match(r'"([^"]+)"\s*->\s*"([^"]+)"', line)
            if edge_match:
                source = edge_match.group(1)
                target = edge_match.group(2)
                
                # Extract edge attributes if present
                attrs = {}
                if '[' in line:
                    attrs_match = re.search(r'\[([^]]+)\]', line)
                    if attrs_match:
                        attrs_str = attrs_match.group(1)
                        # Parse simple attributes
                        for attr_pair in attrs_str.split(';'):
                            if '=' in attr_pair:
                                key, value = attr_pair.split('=', 1)
                                attrs[key.strip()] = value.strip()
                
                graph.add_edge(source, target, **attrs)
    
    return graph


def get_node_label(graph, node):
    """
    Extract the label from a node's attributes.
    """
    if 'label' in graph.nodes[node]:
        return graph.nodes[node]['label']
    return ""


def extract_tensor_size(label):
    """
    Extract tensor size from a node label.
    Tensor sizes are defined by numbers in brackets, e.g., [4096, 512] or [128, 32, 512].
    
    Args:
        label: Node label string
        
    Returns:
        List of integers representing tensor dimensions, or None if not found
    """
    # Pattern to match tensor sizes like [4096, 512] or [128, 32, 512]
    match = re.search(r'\[(\d+(?:,\s*\d+)*)\]', label)
    if match:
        # Extract the numbers and convert to list of integers
        size_str = match.group(1)
        return [int(x.strip()) for x in size_str.split(',')]
    return None


def update_node_label_with_tensor_size(label, new_tensor_size):
    """
    Update a node label with a new tensor size.
    
    Args:
        label: Original node label string
        new_tensor_size: List of integers representing new tensor dimensions
        
    Returns:
        Updated label string with new tensor size
    """
    if new_tensor_size is None:
        return label
    
    # Format the new tensor size
    new_size_str = '[' + ', '.join(str(x) for x in new_tensor_size) + ']'
    
    # Replace the existing tensor size in the label
    updated_label = re.sub(r'\[\d+(?:,\s*\d+)*\]', new_size_str, label, count=1)
    
    return updated_label


def extract_node_name_pattern(label):
    """
    Extract the base name pattern from a node label (e.g., 'Kcur-0', 'kq-0').
    Returns the pattern and layer number.
    Also handles blk.<number> and cache_[kv]_l<number> patterns, returning (None, layer_number).
    """
    # Pattern to match names like "Kcur-0", "kq-15", "Vcur-23", etc.
    match = re.match(r'([A-Za-z_]+)-(\d+)', label)
    if match:
        return match.group(1), int(match.group(2))
    
    # If first pattern doesn't match, try blk.<number> pattern
    blk_match = re.search(r'blk\.(\d+)\.', label)
    if blk_match:
        return None, int(blk_match.group(1))
    
    # If blk pattern doesn't match, try cache_k_l<number> and cache_v_l<number> patterns
    cache_match = re.search(r'cache_[kv]_l(\d+)', label)
    if cache_match:
        return cache_match.group(0), int(cache_match.group(1))

    # If blk/cache_k/v_l pattern doesn't match, try convert_unary_l<number> patterns patterns
    convert_match = re.search(r'convert_unary_l(\d+)', label)
    if convert_match:
        return None, int(convert_match.group(1))
    
    return None, None


def should_remove_node_by_layer(label, max_layers):
    """
    Check if a node should be removed based on its layer number.
    Returns True if the layer number is greater than max_layers.
    Handles patterns like:
    - 'Kcur-15', 'kq-23' (node-<number>)
    - 'blk.15.attn_q.weight' (blk.<number>)
    - 'cache_k_l15', 'cache_v_l23' (cache_k_l<number>, cache_v_l<number>)
    """
    if max_layers is None:
        return False
    
    # Extract layer number from label using the existing pattern (e.g., 'Kcur-15')
    _, layer_num = extract_node_name_pattern(label)
    
    if layer_num is not None and layer_num > max_layers:
        return True

    return False


def is_operation_node(graph, node):
    """
    Check if a node is an operation node.
    Operation nodes have at least one parent and have <x> after the tensor size in the label.
    Data nodes (CONST nodes) start with <x> but don't have operation after tensor size.
    
    Args:
        graph: NetworkX DiGraph
        node: Node ID
        
    Returns:
        Boolean indicating if node is an operation node
    """
    # Check if node has at least one parent
    if graph.in_degree(node) == 0:
        return False
    
    label = get_node_label(graph, node)
    
    # Check if label doesn't start with "<x>" pattern (operation after tensor size)
    # This distinguishes operation nodes from CONST data nodes
    # Check if label starts with <x> (CONST data nodes)
    if label.startswith('<x>'):
        return False
    return True


def extract_operation_from_label(label):
    """
    Extract the operation string from a node label.
    The operation is the string after '| <x>' in the label.
    
    Args:
        label: Node label string
        
    Returns:
        Operation string or None if not found
    """
    # Extract just the operation part, before any newlines (for input/output info)
    match = re.search(r'\|\s*<x>([^\n]+)', label)
    if match:
        return match.group(1).strip()
    return None


def extract_data_type(label):
    """
    Extract the data type from a node label.
    The data type is the value in parentheses, e.g., (f32), (bf16), (f16), (i32), etc.
    
    Args:
        label: Node label string
        
    Returns:
        Data type string or None if not found
    """
    match = re.search(r'\(([^)]+)\)', label)
    if match:
        return match.group(1).strip()
    return None


def update_node_data_type(label, new_data_type):
    """
    Update a node label with a new data type.
    
    Args:
        label: Original node label string
        new_data_type: New data type string
        
    Returns:
        Updated label string with new data type
    """
    # Replace the first occurrence of (data_type) with the new data type
    updated_label = re.sub(r'\([^)]+\)', f'({new_data_type})', label, count=1)
    return updated_label


def extract_node_name(label):
    """
    Extract the node name from a label (the part before the data type).
    
    Args:
        label: Node label string
        
    Returns:
        Node name string
    """
    match = re.match(r'([^\(]+)', label)
    if match:
        return match.group(1).strip()
    return "node"


def add_input_output_to_operation_nodes(graph):
    """
    For each operation node, add input and output information to the label.
    Operation nodes have at least one parent and have <x> after tensor size.
    
    For each operation node:
    - Collect parent tensor sizes
    - Collect own tensor size
    - Add new lines: "input: <parent tensor sizes>" and "output: <node tensor size>"
    - Add formatted "tensor_data" based on operation type
    """
    nodes_modified = 0
    
    for node in graph.nodes():
        if is_operation_node(graph, node):
            label = get_node_label(graph, node)
            node_name = extract_node_name(label)
            node_data_type = extract_data_type(label)
            _, layer_num = extract_node_name_pattern(label)
            
            # Extract operation
            operation = extract_operation_from_label(label)
            if not operation:
                continue
            
            # Extract node's own tensor size
            node_tensor_size = extract_tensor_size(label)
            
            # Collect parent tensor sizes
            parent_tensor_sizes = []
            for parent in graph.predecessors(node):
                parent_label = get_node_label(graph, parent)
                parent_size = extract_tensor_size(parent_label)
                if parent_size:
                    parent_tensor_sizes.append(parent_size)
            
            # Format tensor sizes as strings
            def format_tensor_size(size):
                if size:
                    return '[' + ', '.join(str(x) for x in size) + ']'
                return 'unknown'
            
            # Build input string (all parent tensor sizes)
            if parent_tensor_sizes:
                input_str = ', '.join(format_tensor_size(size) for size in parent_tensor_sizes)
            else:
                input_str = 'none'
            
            # Build output string
            output_str = format_tensor_size(node_tensor_size) if node_tensor_size else 'unknown'
            
            # Build tensor_data string based on operation type
            tensor_data_str = ""
            
            # Format tensor sizes without square brackets for tensor_data
            def format_tensor_data(size):
                if size:
                    return ', '.join(str(x) for x in size)
                return 'unknown'
            
            # Operations that use simple format: (input tensor sizes),(output tensor sizes)
            simple_format_ops = ['x+y', 'set_rows(x)', 'get_rows(x)', 'x*y', 'rms_norm(x)', 'convert_unary', 'rope(x)', 
                                'cont(x)', 'glu(x)', 'soft_max(x)']
            
            if operation in simple_format_ops:
                # Build input string without brackets
                output_data_str = format_tensor_data(node_tensor_size) if node_tensor_size else 'unknown'
                tensor_data_str = f'\ntensor_data: ({output_data_str}),({output_data_str})'
            elif operation == 'X*Y':
                # Calculate B, M, N, K for X*Y operation
                # Check if all parent tensors are 2D or 3D
                # Debug print to see operation and parent tensor sizes
                if parent_tensor_sizes and len(parent_tensor_sizes) >= 2:
                    # Check dimensions of first two parent tensors
                    tensor1_dims = len(parent_tensor_sizes[0])
                    tensor2_dims = len(parent_tensor_sizes[1])
                    
                    # Only process if both tensors are 2D
                    if tensor1_dims == 2 and tensor2_dims == 2:

                        # For 2D tensors: B = 1
                        B = 1
                        
                        # Find common dimension K (the dimension that appears in both tensors)
                        tensor1 = parent_tensor_sizes[0]  # e.g., [M, K]
                        tensor2 = parent_tensor_sizes[1]  # e.g., [K, N]
                        
                        # K is the common number between the two tensors
                        # tensor1[1] should equal tensor2[0] for matrix multiplication
                        K = None
                        M = None
                        N = None
                        # Find the common value between the two tensors
                        common_values = set(tensor1) & set(tensor2)
                        if common_values:
                            K = common_values.pop()
                        # Assign M and N based on which positions K was found
                        if K is not None:
                            # Get all values from both tensors excluding K
                            removed_once = False
                            removed_twice = False
                            remaining_values = []
                            for x in (tensor1 + tensor2):
                                if x == K:
                                    if not removed_once:
                                        removed_once = True
                                        continue
                                    if not removed_twice:
                                        removed_twice = True
                                        continue
                                remaining_values.append(x)
                            assert len(remaining_values) == 2, f"Expected 2 remaining values for X*Y operation, got {len(remaining_values)}: {remaining_values}"
                            M = remaining_values[0]
                            N = remaining_values[1]
                        
                        # Verify M and N are in the output tensor
                        if K is not None and M is not None and N is not None and node_tensor_size:
                            # Assert M and N are in output tensor sizes
                            if len(node_tensor_size) == 2 and M in node_tensor_size and N in node_tensor_size:
                                tensor_data_str = f'\ntensor_data: ({B}), ({M}, {N}, {K})'
                            else:
                                # Assertion failed or output not as expected
                                tensor_data_str = '\ntensor_data: '
                        else:
                            tensor_data_str = '\ntensor_data: '
                    else:
                        # 3D or mixed dimensions - leave blank for now
                        tensor_data_str = '\ntensor_data: '
                        # 3D or mixed dimensions
                        # For 3D tensors, B is the first dimension of the output tensor
                        if len(node_tensor_size) == 3:
                            B = node_tensor_size[2]  # B is the third dimension of output tensor
                            # For 3D matrix multiplication: [B, M, K] x [B, K, N] = [B, M, N]
                            M = node_tensor_size[0]
                            N = node_tensor_size[1]
                            # Find K from the 3D tensor structure
                            # In 3D matrix multiplication, B is typically in the last position of output
                            # For tensors like [M, K, B] x [K, N, B] = [M, N, B]
                            # We need to find which parent tensor has B (dimension of 3) and extract K
                            K = None
                            for parent_tensor in parent_tensor_sizes:
                                if len(parent_tensor) >= 3:
                                    if parent_tensor[0] in [M, N]:
                                        K = parent_tensor[1]
                                        break
                                    else:
                                        K = parent_tensor[0]
                                        break
                                else:
                                    K = None
                            if K is not None:
                                tensor_data_str = f'\ntensor_data: ({B}), ({M}, {N}, {K})'
                        else:
                            tensor_data_str = '\ntensor_data: '
                else:
                    tensor_data_str = '\ntensor_data: '
            
            # Add new lines to label
            # The label format is: "name (type)|number [tensor_size] | <x>operation"
            # We want to add after the operation
            # new_label = f"operation: {operation}\ndata_type: {node_data_type}\ninput: {input_str}\noutput: {output_str}{tensor_data_str}"
            new_label = f"node_name: {node_name}\nlayer: {layer_num}\noperation: {operation}\ndata_type: {node_data_type}{tensor_data_str}"
            
            # new_label = label + f'\ninput: {input_str}\noutput: {output_str}{tensor_data_str}'
            
            # Update the node label
            graph.nodes[node]['old_label'] = graph.nodes[node]['label']
            graph.nodes[node]['label'] = new_label
            nodes_modified += 1
    
    print(f"Modified {nodes_modified} operation nodes with input/output information")


def insert_conversion_nodes_for_mixed_types(graph):
    """
    For operation nodes with mixed parent data types (not all f32), insert intermediate conversion nodes.
    Only applies to nodes with X*Y operation.
    
    For each operation node with X*Y operation:
    1. Check if parent nodes have different data types
    2. If a parent is f32 and another parent is not f32:
       - Insert convert_unary node between f32 parent and operation node
       - Insert convert_unary node between operation node and its children
       - Update operation node's data type to the non-f32 type
    """
    conversion_counter = 0
    nodes_to_process = []
    
    # First pass: identify nodes that need conversion
    for node in list(graph.nodes()):
        if not is_operation_node(graph, node):
            continue
        
        label = get_node_label(graph, node)
        
        # Check if the operation is X*Y
        operation = extract_operation_from_label(label)
        if operation != 'X*Y':
            continue
        
        node_data_type = extract_data_type(label)
        
        # Get parent data types
        parent_types = []
        parents = list(graph.predecessors(node))
        
        for parent in parents:
            parent_label = get_node_label(graph, parent)
            parent_type = extract_data_type(parent_label)
            if parent_type:
                parent_types.append((parent, parent_type))
        
        # Check if there are mixed types (f32 and non-f32)
        has_f32 = any(ptype == 'f32' for _, ptype in parent_types)
        has_non_f32 = any(ptype != 'f32' for _, ptype in parent_types)
        
        if has_f32 and has_non_f32:
            # Find the non-f32 type
            other_type = next((ptype for _, ptype in parent_types if ptype != 'f32'), None)
            if other_type:
                nodes_to_process.append((node, parent_types, other_type))
    
    # Second pass: insert conversion nodes
    for node, parent_types, other_type in nodes_to_process:
        label = get_node_label(graph, node)
        _, layer_num = extract_node_name_pattern(label)
        node_tensor_size = extract_tensor_size(label)
        node_name = extract_node_name(label)
        
        # Insert conversion nodes between f32 parents and the operation node
        for parent, parent_type in parent_types:
            if parent_type == 'f32':
                parent_label = get_node_label(graph, parent)
                parent_tensor_size = extract_tensor_size(parent_label)
                
                # Create intermediate conversion node (f32 -> other_type)
                conv_node_id = f"conv_{conversion_counter}_in"
                conversion_counter += 1
                
                # Create label for conversion node
                conv_label = f"convert_unary_l{layer_num} ({parent_type}, {other_type})|{conversion_counter} {format_tensor_size(parent_tensor_size)} | <x>convert_unary"
                
                # Add the conversion node
                graph.add_node(conv_node_id, label=conv_label)
                
                # Get edge attributes from parent to operation node (check if edge exists)
                if graph.has_edge(parent, node):
                    edge_attrs = graph.get_edge_data(parent, node)
                    
                    # Remove edge from parent to operation node
                    graph.remove_edge(parent, node)
                    
                    # Add edge from parent to conversion node
                    if edge_attrs:
                        graph.add_edge(parent, conv_node_id, **edge_attrs)
                    else:
                        graph.add_edge(parent, conv_node_id)
                    
                    # Add edge from conversion node to operation node
                    graph.add_edge(conv_node_id, node)
                    
                    print(f"Inserted conversion node {conv_node_id} between {parent_label.split('|')[0]} and {label.split('|')[0]}")
        
        # Update the operation node's data type to other_type
        updated_label = update_node_data_type(label, other_type)
        graph.nodes[node]['label'] = updated_label
        
        # Insert conversion node between operation node and its children (other_type -> f32)
        children = list(graph.successors(node))
        
        if children:
            # Create output conversion node
            conv_out_node_id = f"conv_{conversion_counter}_out"
            conversion_counter += 1
            
            # Create label for output conversion node
            conv_out_label = f"convert_unary_l{layer_num} ({other_type}, f32)|{conversion_counter} {format_tensor_size(node_tensor_size)} | <x>convert_unary"
            
            # Add the output conversion node
            graph.add_node(conv_out_node_id, label=conv_out_label)
            
            # Redirect edges from operation node to children through conversion node
            for child in children:
                edge_attrs = graph.get_edge_data(node, child)
                
                # Remove edge from operation node to child
                graph.remove_edge(node, child)
                
                # Add edge from operation node to conversion node
                graph.add_edge(node, conv_out_node_id)
                
                # Add edge from conversion node to child
                if edge_attrs:
                    graph.add_edge(conv_out_node_id, child, **edge_attrs)
                else:
                    graph.add_edge(conv_out_node_id, child)
            
            print(f"Inserted conversion node {conv_out_node_id} between {updated_label.split('|')[0]} and its children")
    
    print(f"Inserted {conversion_counter} conversion nodes for mixed data types")


def remove_non_operation_nodes(graph):
    """
    Remove all nodes that are not operation nodes, while connecting parents to children.
    Operation nodes are identified by the is_operation_node function.
    """
    nodes_to_remove = []
    
    # Identify all non-operation nodes
    for node in list(graph.nodes()):
        if not is_operation_node(graph, node):
            nodes_to_remove.append(node)
    
    print(f"Found {len(nodes_to_remove)} non-operation nodes to remove")
    
    # For each non-operation node, connect its parents to its children
    for node in nodes_to_remove:
        # Get predecessors (parents) and successors (children)
        parents = list(graph.predecessors(node))
        children = list(graph.successors(node))
        
        # Connect each parent to each child
        for parent in parents:
            parent_edge_attrs = graph.get_edge_data(parent, node)
            
            for child in children:
                child_edge_attrs = graph.get_edge_data(node, child)
                
                # Create new edge from parent to child
                # Use parent edge attributes if available
                new_attrs = {}
                if parent_edge_attrs:
                    new_attrs.update(parent_edge_attrs)
                
                graph.add_edge(parent, child, **new_attrs)
    
    # Remove all non-operation nodes
    graph.remove_nodes_from(nodes_to_remove)
    
    print(f"Removed {len(nodes_to_remove)} non-operation nodes")
    print(f"Remaining nodes: {graph.number_of_nodes()}")
    print(f"Remaining edges: {graph.number_of_edges()}")


def print_topological_sort(graph):
    """
    Perform topological sort on the graph and print node names in dependency order.
    
    Args:
        graph: NetworkX DiGraph containing operation nodes
    """
    try:
        print("\n" + "="*60)
        print("Topological Sort of Operation Nodes:")
        print("="*60)
        
        topo_order = list(nx.topological_sort(graph))
        
        for idx, node in enumerate(topo_order, 1):
            label = get_node_label(graph, node)
            node_name = extract_node_name(label)
            operation = extract_operation_from_label(label)
            
            # Extract layer information
            _, layer_num = extract_node_name_pattern(label)
            
            # Print node information
            if operation:
                print(f"{idx:4d}. {node_name:30s} | Operation: {operation:20s} | Layer: {layer_num}")
            else:
                print(f"{idx:4d}. {node_name:30s} | Layer: {layer_num}")
        
        print("="*60)
        print(f"Total operation nodes in topological order: {len(topo_order)}")
        print("="*60 + "\n")
        
    except nx.NetworkXError as e:
        print(f"Error: Graph contains a cycle, cannot perform topological sort: {e}")


def print_dfs_traversal(graph):
    """
    Perform DFS (Depth-First Search) traversal on the graph and print node names.
    Starts from all source nodes (nodes with no incoming edges).
    
    Args:
        graph: NetworkX DiGraph containing operation nodes
    """
    print("\n" + "="*60)
    print("DFS (Depth-First Search) Traversal of Operation Nodes:")
    print("="*60)
    
    # Find all source nodes (nodes with no incoming edges)
    source_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    
    if not source_nodes:
        print("Warning: No source nodes found. Graph may contain cycles.")
        # If no source nodes, just pick the first node
        if graph.number_of_nodes() > 0:
            source_nodes = [list(graph.nodes())[0]]
    
    visited = set()
    dfs_order = []
    
    # Manual DFS implementation using a stack
    def dfs_visit(node):
        """Recursive DFS helper function."""
        if node in visited:
            return
        visited.add(node)
        dfs_order.append(node)
        
        # Visit all successors
        for neighbor in graph.successors(node):
            if neighbor not in visited:
                dfs_visit(neighbor)
    
    # Perform DFS from each source node
    for source in source_nodes:
        if source not in visited:
            dfs_visit(source)
    
    # Print nodes in DFS order
    for idx, node in enumerate(dfs_order, 1):
        label = get_node_label(graph, node)
        node_name = extract_node_name(label)
        operation = extract_operation_from_label(label)
        
        # Extract layer information
        _, layer_num = extract_node_name_pattern(label)
        
        # Print node information
        if operation:
            print(f"{idx:4d}. {node_name:30s} | Operation: {operation:20s} | Layer: {layer_num}")
        else:
            print(f"{idx:4d}. {node_name:30s} | Layer: {layer_num}")
    
    print("="*60)
    print(f"Total operation nodes in DFS order: {len(dfs_order)}")
    print(f"Source nodes: {len(source_nodes)}")
    print("="*60 + "\n")


def print_bfs_traversal(graph):
    """
    Perform BFS (Breadth-First Search) traversal on the graph and print node names.
    Starts from all source nodes (nodes with no incoming edges).
    
    Args:
        graph: NetworkX DiGraph containing operation nodes
    """
    print("\n" + "="*60)
    print("BFS (Breadth-First Search) Traversal of Operation Nodes:")
    print("="*60)
    
    # Find all source nodes (nodes with no incoming edges)
    source_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    
    if not source_nodes:
        print("Warning: No source nodes found. Graph may contain cycles.")
        # If no source nodes, just pick the first node
        if graph.number_of_nodes() > 0:
            source_nodes = [list(graph.nodes())[0]]
    
    visited = set()
    bfs_order = []
    
    # Use a queue for BFS
    queue = deque(source_nodes)
    
    # Mark all source nodes as visited
    for source in source_nodes:
        visited.add(source)
        bfs_order.append(source)
    
    # Perform BFS
    while queue:
        current = queue.popleft()
        
        # Visit all neighbors
        for neighbor in graph.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                bfs_order.append(neighbor)
                queue.append(neighbor)
    
    # Print nodes in BFS order
    for idx, node in enumerate(bfs_order, 1):
        label = get_node_label(graph, node)
        node_name = extract_node_name(label)
        operation = extract_operation_from_label(label)
        
        # Extract layer information
        _, layer_num = extract_node_name_pattern(label)
        
        # Print node information
        if operation:
            print(f"{idx:4d}. {node_name:30s} | Operation: {operation:20s} | Layer: {layer_num}")
        else:
            print(f"{idx:4d}. {node_name:30s} | Layer: {layer_num}")
    
    print("="*60)
    print(f"Total operation nodes in BFS order: {len(bfs_order)}")
    print(f"Source nodes: {len(source_nodes)}")
    print("="*60 + "\n")

def get_chain_of_children(graph, node, in_degree):
    """
    Get the chain of children for a given node.
    """
    children = list(graph.successors(node))
    if len(children) != 1 or in_degree[children[0]] > 1 or not children:
        return []
    return [node] + get_chain_of_children(graph, children[0], in_degree)

def print_dependency_order_traversal(graph):
    """
    Perform dependency-order traversal with depth-first priority.
    Only visits a node when ALL of its parents have been visited.
    When a child becomes ready (all parents visited), it's added to the FRONT
    of the queue, creating a depth-first exploration pattern.
    
    Args:
        graph: NetworkX DiGraph containing operation nodes
    """
    print("\n" + "="*60)
    print("Dependency-Order Traversal (All Parents First, Depth-First Priority):")
    print("="*60)
    
    # Track in-degree for each node (number of unvisited parents)
    in_degree = {node: graph.in_degree(node) for node in graph.nodes()}
    
    # Find all source nodes (nodes with no incoming edges)
    ready_queue = deque([node for node, degree in in_degree.items() if degree == 0])
    
    visited = set()
    dependency_order = []
    
    if not ready_queue:
        print("Warning: No source nodes found. Graph may contain cycles.")
        print("="*60 + "\n")
        return
    
    # Process nodes in dependency order with depth-first priority
    while ready_queue:
        # Get the next node whose all parents have been visited
        current = ready_queue.popleft()
        
        if current not in visited:
            # Visit this node
            visited.add(current)
            in_degree[current] = max(0, in_degree[current] - 1)
            for succ in graph.successors(current):
                if succ in in_degree:
                    in_degree[succ] = max(0, in_degree[succ] - 1)
            dependency_order.append(current)
            
            chain = get_chain_of_children(graph, current, in_degree)
            for node in chain:
                visited.add(node)
                # INSERT_YOUR_CODE
                # Reduce the in_degree of node (simulate "visiting" its parent)
                if node != current:
                    in_degree[node] = max(0, in_degree[node] - 1)
                    # If node has no unvisited parents, add it to ready queue
                    assert in_degree[node] == 0
                    dependency_order.append(node)
                for succ in graph.successors(node):
                    if succ in in_degree:
                        in_degree[succ] = max(0, in_degree[succ] - 1)
        ready_queue = deque([node for node, degree in in_degree.items() if degree == 0 and node not in visited])

    
    for node in dependency_order:
        # Print node information
        # Fix: Get node attributes for operation, node_name, and layer_num per node.
        node_label = get_node_label(graph, node) if 'get_node_label' in globals() else str(node)
        operation = extract_operation_from_label(node_label) if 'extract_operation_from_label' in globals() else ""
        node_name = extract_node_name(node_label) if 'extract_node_name' in globals() else str(node)
        _, layer_num = extract_node_name_pattern(node_label) if 'extract_node_name_pattern' in globals() else ""
        idx = dependency_order.index(node) + 1

        if operation:
            print(f"{idx:4d}. {node_name:30s} | Operation: {operation:20s} | Layer: {layer_num}")
        else:
            print(f"{idx:4d}. {node_name:30s} | Layer: {layer_num}")
    
    print("="*60)
    print(f"Total operation nodes in dependency order: {len(dependency_order)}")
    print("="*60 + "\n")


def format_tensor_size(size):
    """Helper function to format tensor size as string."""
    if size:
        return '[' + ', '.join(str(x) for x in size) + ']'
    return '[unknown]'


def process_graph(graph, max_layers=None):
    """
    Process the graph to remove nodes with 'view(', 'reshape(', or 'permute(' in their labels.
    Connect parent nodes directly to child nodes when removing a node.
    Also add edges from childless Kcur-* to kq-* and childless Vcur-* to kqv-*.
    Optionally filter nodes by layer number if max_layers is specified.
    """
    nodes_to_remove = []
    
    # Identify nodes to remove based on keywords
    for node in graph.nodes():
        label = get_node_label(graph, node)
        if should_remove_node(label):
            nodes_to_remove.append(node)
    
    print(f"Found {len(nodes_to_remove)} nodes to remove (view(/reshape(/permute()")
    
    # Identify nodes to remove based on layer number
    if max_layers is not None:
        layer_nodes_to_remove = []
        for node in graph.nodes():
            label = get_node_label(graph, node)
            if should_remove_node_by_layer(label, max_layers):
                layer_nodes_to_remove.append(node)
        
        print(f"Found {len(layer_nodes_to_remove)} nodes to remove (layer > {max_layers})")
        nodes_to_remove.extend(layer_nodes_to_remove)
        nodes_to_remove = list(set(nodes_to_remove))  # Remove duplicates
    
    print(f"Total nodes to remove: {len(nodes_to_remove)}")
    
    # For each node to remove, connect its parents to its children
    # and update parent node labels with the removed node's tensor size
    for node in nodes_to_remove:
        # Get the tensor size of the node being removed
        node_label = get_node_label(graph, node)
        removed_node_tensor_size = extract_tensor_size(node_label)
        
        # Get predecessors (parents) and successors (children)
        parents = list(graph.predecessors(node))
        children = list(graph.successors(node))
        
        # Get edge attributes from parent->node and node->child edges
        for parent in parents:
            parent_edge_attrs = graph.get_edge_data(parent, node)
            
            # Update parent node's label with the removed node's tensor size
            # Only do this if the parent node doesn't contain "X*Y"
            if removed_node_tensor_size is not None:
                parent_label = get_node_label(graph, parent)
                if 'X*Y' not in parent_label:
                    updated_parent_label = update_node_label_with_tensor_size(parent_label, removed_node_tensor_size)
                    graph.nodes[parent]['label'] = updated_parent_label
            
            for child in children:
                child_edge_attrs = graph.get_edge_data(node, child)
                
                # Create new edge from parent to child
                # Combine attributes if needed, or use parent edge attributes
                new_attrs = {}
                if parent_edge_attrs:
                    new_attrs.update(parent_edge_attrs)
                
                graph.add_edge(parent, child, **new_attrs)
    
    # Remove the identified nodes
    graph.remove_nodes_from(nodes_to_remove)
    
    print(f"Removed {len(nodes_to_remove)} nodes")
    print(f"Remaining nodes: {graph.number_of_nodes()}")
    print(f"Remaining edges: {graph.number_of_edges()}")
    
    # Reverse edges for cache_v_l* and cache_k_l* nodes with set_rows
    reverse_cache_set_rows_edges(graph)
    
    # Insert conversion nodes for mixed data types AFTER removing nodes
    insert_conversion_nodes_for_mixed_types(graph)
    
    # Add input/output labels to newly created conversion nodes
    add_input_output_to_operation_nodes(graph)
    
    # Remove all non-operation nodes while connecting parents to children
    remove_non_operation_nodes(graph)

    # # Add edges from childless Kcur-* to kq-* and childless Vcur-* to kqv-*
    add_missing_edges(graph)
    
    # # Perform different graph traversals and print node names
    # # print_topological_sort(graph)
    # # print_dfs_traversal(graph)
    # # print_bfs_traversal(graph)
    print_dependency_order_traversal(graph)
    
    return graph


def reverse_cache_set_rows_edges(graph):
    """
    Reverse edges between cache_v_l*/cache_k_l* nodes with set_rows and their parents
    that also have cache_v_l*/cache_k_l*.
    """
    edges_to_reverse = []
    
    for node in graph.nodes():
        label = get_node_label(graph, node)
        
        # Check if this node has cache_v_l* and set_rows
        if 'cache_v_l' in label and 'set_rows' in label:
            # Find parents that also have cache_v_l*
            for parent in graph.predecessors(node):
                parent_label = get_node_label(graph, parent)
                if 'cache_v_l' in parent_label:
                    edges_to_reverse.append((parent, node))
                    print(f"Found cache_v edge to reverse: {parent_label.split('|')[0]} -> {label.split('|')[0]}")
        
        # Check if this node has cache_k_l* and set_rows
        if 'cache_k_l' in label and 'set_rows' in label:
            # Find parents that also have cache_k_l*
            for parent in graph.predecessors(node):
                parent_label = get_node_label(graph, parent)
                if 'cache_k_l' in parent_label:
                    edges_to_reverse.append((parent, node))
                    print(f"Found cache_k edge to reverse: {parent_label.split('|')[0]} -> {label.split('|')[0]}")
    
    # Reverse the edges
    for parent, child in edges_to_reverse:
        # Get edge attributes
        edge_attrs = graph.get_edge_data(parent, child)
        
        # Remove the old edge
        graph.remove_edge(parent, child)
        
        # Add the reversed edge
        if edge_attrs:
            graph.add_edge(child, parent, **edge_attrs)
        else:
            graph.add_edge(child, parent)
        
        parent_label = get_node_label(graph, parent)
        child_label = get_node_label(graph, child)
        print(f"Reversed edge: {child_label.split('|')[0]} -> {parent_label.split('|')[0]}")
    
    print(f"Reversed {len(edges_to_reverse)} edges")


def add_missing_edges(graph):
    """
    Add edges from childless Kcur-* nodes to corresponding kq-* nodes,
    and from childless Vcur-* nodes to corresponding kqv-* nodes.
    """
    # Build a mapping of node patterns to nodes
    kcur_cache_nodes = {}  # layer -> node_id
    kq_nodes = {}    # layer -> node_id
    vcur_cache_nodes = {}  # layer -> node_id
    kqv_nodes = {}   # layer -> node_id
    
    for node in graph.nodes():
        label = graph.nodes[node]['old_label']
        pattern, layer = extract_node_name_pattern(label)
        
        if pattern is not None and 'cache_k_l' in pattern and layer is not None:
            if layer not in kcur_cache_nodes:
                kcur_cache_nodes[layer] = []
            kcur_cache_nodes[layer].append(node)
        elif pattern == 'kq' and layer is not None:
            kq_nodes[layer] = node
        elif pattern is not None and 'cache_v_l' in pattern and layer is not None:
            if layer not in vcur_cache_nodes:
                vcur_cache_nodes[layer] = []
            vcur_cache_nodes[layer].append(node)
        elif pattern == 'kqv' and layer is not None:
            kqv_nodes[layer] = node
    
    edges_added = 0
    
    # Connect childless Kcur-* to kq-*
    for layer, kcur_list in kcur_cache_nodes.items():
        for kcur_node in kcur_list:
            # Check if this Kcur node has no children
            if graph.out_degree(kcur_node) == 0:
                # Find corresponding kq node
                if layer in kq_nodes:
                    kq_node = kq_nodes[layer]
                    graph.add_edge(kcur_node, kq_node, arrowhead='vee', style='solid', label=f'src 0')
                    edges_added += 1
                    print(f"Added edge: {get_node_label(graph, kcur_node).split('|')[0]} -> {get_node_label(graph, kq_node).split('|')[0]}")
    
    # Connect childless Vcur-* to kq-*
    for layer, vcur_list in vcur_cache_nodes.items():
        for vcur_node in vcur_list:
            # Check if this Vcur node has no children
            # if graph.out_degree(vcur_node) == 0:
                # Find corresponding kq node
                if layer in kq_nodes:
                    kq_node = kq_nodes[layer]
                    graph.add_edge(vcur_node, kq_node, arrowhead='vee', style='solid', label=f'src 2')
                    edges_added += 1
                    print(f"Added edge: {get_node_label(graph, vcur_node).split('|')[0]} -> {get_node_label(graph, kq_node).split('|')[0]}")
    
    print(f"Added {edges_added} missing edges")


def write_dot_file(graph, filepath):
    """
    Write the processed graph to a DOT file.
    """
    try:
        nx.nx_pydot.write_dot(graph, filepath)
        print(f"Written processed graph to {filepath}")
    except Exception as e:
        print(f"Error writing {filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Process llama_*.dot files to remove view(/reshape(/permute( nodes'
    )
    parser.add_argument(
        'input_pattern',
        nargs='?',
        default='llama_*.dot',
        help='Pattern to match input DOT files (default: llama_*.dot)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for processed files (default: current directory)'
    )
    parser.add_argument(
        '--suffix',
        default='_processed',
        help='Suffix to add to output filenames (default: _processed)'
    )
    parser.add_argument(
        '--max-layers',
        type=int,
        default=None,
        help='Maximum layer number to include. Nodes with layer number greater than this will be removed (optional)'
    )
    
    args = parser.parse_args()
    
    # Find all matching files
    input_files = glob.glob(args.input_pattern)
    
    if not input_files:
        print(f"No files matching pattern '{args.input_pattern}' found")
        return
    
    print(f"Found {len(input_files)} file(s) to process")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print('='*60)
        
        # Parse the DOT file
        graph = parse_dot_file(input_file)
        if graph is None:
            continue
        
        print(f"Original graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Process the graph
        processed_graph = process_graph(graph, max_layers=args.max_layers)
        
        # Generate output filename
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(
            args.output_dir,
            f"{name_without_ext}{args.suffix}.dot"
        )
        
        # Write the processed graph
        write_dot_file(processed_graph, output_file)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print('='*60)


if __name__ == '__main__':
    main()
