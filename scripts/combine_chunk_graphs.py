#!/usr/bin/env python3
"""
Combine multiple DOT files for chunk-level parallelism using NetworkX.
Each input file represents a chunk that can be executed in parallel.
"""

import re
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple
import sys


class ChunkGraph:
    """Represents a single chunk graph."""
    
    def __init__(self, chunk_idx: int, filepath: str):
        self.chunk_idx = chunk_idx
        self.filepath = filepath
        self.graph = None
        self.load_graph()
    
    def load_graph(self):
        """Load the DOT file into a NetworkX graph."""
        try:
            # Read as directed graph
            self.graph = nx.DiGraph(nx.drawing.nx_pydot.read_dot(self.filepath))
            print(f"Loaded chunk {self.chunk_idx}: {Path(self.filepath).name}")
            print(f"  Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        except Exception as e:
            print(f"Error loading {self.filepath}: {e}")
            raise
    
    def get_input_nodes(self) -> List[str]:
        """Get nodes with no incoming edges (entry points)."""
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
    
    def get_output_nodes(self) -> List[str]:
        """Get nodes with no outgoing edges (exit points)."""
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]
    
    def rename_nodes(self, prefix: str) -> nx.DiGraph:
        """Create a copy of the graph with renamed nodes."""
        mapping = {node: f"{prefix}{node}" for node in self.graph.nodes()}
        renamed_graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        
        # Update node labels to include chunk information
        for node in renamed_graph.nodes():
            if 'label' in renamed_graph.nodes[node]:
                original_label = renamed_graph.nodes[node]['label']
                # Remove quotes if present
                if original_label.startswith('"') and original_label.endswith('"'):
                    original_label = original_label[1:-1]
                renamed_graph.nodes[node]['label'] = f'"chunk: {self.chunk_idx}\\n{original_label}"'
        
        return renamed_graph, mapping


class ChunkParallelCombiner:
    """Combines multiple chunk graphs into a single parallel graph."""
    
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.chunks: List[ChunkGraph] = []
        self.combined_graph = nx.DiGraph()
        self.chunk_mappings = []  # Store node mappings for each chunk
    
    def load_chunks(self, pattern: str = "llama_*_examen.dot"):
        """Load all chunk files matching the pattern."""
        files = sorted(self.input_dir.glob(pattern))
        
        if not files:
            print(f"No files found matching pattern: {pattern} in {self.input_dir}")
            return
        
        for filepath in files:
            # Extract chunk index from filename
            match = re.search(r'llama_(\d+)_examen\.dot', filepath.name)
            if match:
                chunk_idx = int(match.group(1))
                chunk = ChunkGraph(chunk_idx, str(filepath))
                self.chunks.append(chunk)
        
        # Sort by chunk index
        self.chunks.sort(key=lambda c: c.chunk_idx)
        print(f"\nTotal chunks loaded: {len(self.chunks)}\n")
    
    def combine_chunks(self):
        """Combine all chunk graphs into a single graph with prefixed node names."""
        print("Combining chunks...")
        
        for chunk in self.chunks:
            # Rename nodes with chunk prefix
            prefix = f"chunk_{chunk.chunk_idx}_"
            renamed_graph, mapping = chunk.rename_nodes(prefix)
            
            # Store mapping for later use
            self.chunk_mappings.append({
                'chunk_idx': chunk.chunk_idx,
                'mapping': mapping,
                'inverse_mapping': {v: k for k, v in mapping.items()},
                'input_nodes': [mapping[n] for n in chunk.get_input_nodes()],
                'output_nodes': [mapping[n] for n in chunk.get_output_nodes()]
            })
            
            # Add all nodes and edges from this chunk to combined graph
            self.combined_graph.add_nodes_from(renamed_graph.nodes(data=True))
            self.combined_graph.add_edges_from(renamed_graph.edges(data=True))
        
        print(f"Combined graph created:")
        print(f"  Total nodes: {self.combined_graph.number_of_nodes()}")
        print(f"  Total edges: {self.combined_graph.number_of_edges()}")
    
    def print_chunk_summary(self):
        """Print summary of chunks and their interface nodes."""
        print("\n" + "="*80)
        print("CHUNK INTERFACE SUMMARY")
        print("="*80)
        
        for mapping_info in self.chunk_mappings:
            chunk_idx = mapping_info['chunk_idx']
            input_nodes = mapping_info['input_nodes']
            output_nodes = mapping_info['output_nodes']
            
            print(f"\n📦 Chunk {chunk_idx}:")
            print(f"  Input nodes (entry points): {len(input_nodes)}")
            for node in input_nodes[:5]:
                # Get node label if available
                label = self.combined_graph.nodes[node].get('label', node)
                if label.startswith('"') and label.endswith('"'):
                    label = label[1:-1]
                # Extract just the node name
                node_name = node.split('_', 2)[-1] if '_' in node else node
                print(f"    • {node_name}")
            if len(input_nodes) > 5:
                print(f"    ... and {len(input_nodes) - 5} more")
            
            print(f"  Output nodes (exit points): {len(output_nodes)}")
            for node in output_nodes[:5]:
                node_name = node.split('_', 2)[-1] if '_' in node else node
                print(f"    • {node_name}")
            if len(output_nodes) > 5:
                print(f"    ... and {len(output_nodes) - 5} more")
    
    def add_inter_chunk_edges(self):
        """
        Add inter-chunk edges for cache dependencies:
        - cache_k_l{layer} (chunk N) -> Kcur-{layer} with X*Y operation (chunk N+1)
        - cache_v_l{layer} (chunk N) -> Vcur-{layer} with X*Y operation (chunk N+1)
        For all layers present in the graphs.
        """
        print("\nAdding inter-chunk edges...")
        
        # First, detect all unique layers in the graph
        layers = self._detect_layers()
        print(f"Detected layers: {sorted(layers)}")
        
        edges_added = 0
        
        for i in range(len(self.chunks) - 1):
            # Get current chunk (N) and next chunk (N+1)
            current_chunk_idx = self.chunks[i].chunk_idx
            next_chunk_idx = self.chunks[i + 1].chunk_idx
            
            print(f"\n  Connecting chunk {current_chunk_idx} → chunk {next_chunk_idx}:")
            
            # For each layer, add inter-chunk edges
            for layer_num in sorted(layers):
                # Find cache_k and cache_v nodes for this layer in current chunk
                cache_k_node = self._find_node_by_pattern(
                    current_chunk_idx, 
                    node_name_pattern=f"cache_k_l{layer_num}",
                    operation_pattern="set_rows(x)",
                    layer_num=layer_num
                )
                cache_v_node = self._find_node_by_pattern(
                    current_chunk_idx,
                    node_name_pattern=f"cache_v_l{layer_num}",
                    operation_pattern="set_rows(x)",
                    layer_num=layer_num
                )
                
                # Find Kcur and Vcur nodes with X*Y operation for this layer in next chunk
                kcur_node = self._find_node_by_pattern(
                    next_chunk_idx,
                    node_name_pattern=f"Kcur-{layer_num}",
                    operation_pattern="X*Y",
                    layer_num=layer_num
                )
                vcur_node = self._find_node_by_pattern(
                    next_chunk_idx,
                    node_name_pattern=f"Vcur-{layer_num}",
                    operation_pattern="X*Y",
                    layer_num=layer_num
                )
                
                # Add edges
                if cache_k_node and kcur_node:
                    self.combined_graph.add_edge(cache_k_node, kcur_node)
                    print(f"    ✓ Layer {layer_num}: {cache_k_node} -> {kcur_node}")
                    edges_added += 1
                else:
                    print(f"    ✗ Layer {layer_num}: Could not find cache_k_l{layer_num} or Kcur-{layer_num} nodes")
                
                if cache_v_node and vcur_node:
                    self.combined_graph.add_edge(cache_v_node, vcur_node)
                    print(f"    ✓ Layer {layer_num}: {cache_v_node} -> {vcur_node}")
                    edges_added += 1
                else:
                    print(f"    ✗ Layer {layer_num}: Could not find cache_v_l{layer_num} or Vcur-{layer_num} nodes")
        
        print(f"\nTotal inter-chunk edges added: {edges_added}")
        return edges_added
    
    def _detect_layers(self):
        """Detect all unique layer numbers in the combined graph."""
        layers = set()
        
        for node in self.combined_graph.nodes():
            label = self.combined_graph.nodes[node].get('label', '')
            
            # Remove quotes if present
            if label.startswith('"') and label.endswith('"'):
                label = label[1:-1]
            
            # Extract layer number from label
            import re
            match = re.search(r'layer:\s*(\d+)', label)
            if match:
                layers.add(int(match.group(1)))
        
        return layers
    
    def _find_node_by_pattern(self, chunk_idx: int, node_name_pattern: str, operation_pattern: str, layer_num: int = None):
        """
        Find a node in the combined graph by matching patterns in its label.
        
        Args:
            chunk_idx: The chunk index to search in
            node_name_pattern: Pattern to match in node_name field
            operation_pattern: Pattern to match in operation field
            layer_num: Optional layer number to match
        
        Returns:
            Node ID if found, None otherwise
        """
        prefix = f"chunk_{chunk_idx}_"
        
        for node in self.combined_graph.nodes():
            # Only look at nodes from the specified chunk
            if not node.startswith(prefix):
                continue
            
            # Get the node's label
            label = self.combined_graph.nodes[node].get('label', '')
            
            # Remove quotes if present
            if label.startswith('"') and label.endswith('"'):
                label = label[1:-1]
            
            # Check if label contains the patterns
            if node_name_pattern in label and operation_pattern in label:
                # Parse the label to verify it's the right node
                if f"node_name: {node_name_pattern}" in label or node_name_pattern in label:
                    if f"operation: {operation_pattern}" in label:
                        # If layer_num is specified, verify it matches
                        if layer_num is not None:
                            if f"layer: {layer_num}" in label:
                                return node
                        else:
                            return node
        
        return None
    
    def save_combined_graph(self):
        """Save the combined graph to a DOT file."""
        print(f"\nSaving combined graph to: {self.output_file}")
        
        # Write using pydot to maintain DOT format
        try:
            nx.drawing.nx_pydot.write_dot(self.combined_graph, str(self.output_file))
            print("✓ Successfully saved combined graph")
        except Exception as e:
            print(f"Error saving graph: {e}")
            raise
        
        # Add comments about inter-chunk edges
        self._add_comments_to_output()
    
    def _add_comments_to_output(self):
        """Add helpful comments to the output DOT file."""
        with open(self.output_file, 'r') as f:
            content = f.read()
        
        # Find the last edge before the closing brace
        lines = content.split('\n')
        
        # Detect layers
        layers = self._detect_layers()
        layer_list = ", ".join(str(l) for l in sorted(layers))
        
        # Insert comments before closing brace
        comments = [
            '',
            '/* ========================================',
            ' * INTER-CHUNK EDGES SUMMARY',
            ' * ========================================',
            ' * Inter-chunk edges have been added for cache dependencies',
            ' * across all layers.',
            ' * ',
            f' * Layers present: {layer_list}',
            ' * ',
            ' * Pattern (for each layer L):',
            ' *   cache_k_lL (chunk N) → Kcur-L X*Y operation (chunk N+1)',
            ' *   cache_v_lL (chunk N) → Vcur-L X*Y operation (chunk N+1)',
            ' * ',
            ' * Chunk connections:',
            ' *   Chunk 0 → Chunk 1',
            ' *   Chunk 1 → Chunk 2',
            ' *   Chunk 2 → Chunk 3',
            f' *   (Chunk {len(self.chunks)-1}\'s cache nodes have no outgoing inter-chunk edges)',
            ' * ',
            ' * Chunk interface nodes:',
        ]
        
        for mapping_info in self.chunk_mappings:
            chunk_idx = mapping_info['chunk_idx']
            comments.append(f' * ')
            comments.append(f' * Chunk {chunk_idx}:')
            comments.append(f' *   Input nodes: {", ".join(mapping_info["input_nodes"])}')
            comments.append(f' *   Output nodes: {", ".join(mapping_info["output_nodes"])}')
        
        comments.append(' */')
        comments.append('')
        
        # Find the closing brace
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == '}':
                # Insert comments before the closing brace
                lines.insert(i, '\n'.join(comments))
                break
        
        # Write back
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(lines))
    
    def get_chunk_interface_info(self):
        """Return detailed information about chunk interfaces."""
        return self.chunk_mappings


def main():
    """Main function."""
    
    # Configuration
    script_dir = Path(__file__).parent
    input_dir = script_dir.parent / "output_dot"
    output_file = input_dir / "combined_chunk_parallel.dot"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    
    print("="*80)
    print("CHUNK-LEVEL PARALLEL DOT COMBINER (NetworkX)")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Create combiner
    combiner = ChunkParallelCombiner(str(input_dir), str(output_file))
    
    # Load chunks
    combiner.load_chunks()
    
    if not combiner.chunks:
        print("Error: No chunks loaded!")
        return 1
    
    # Combine chunks
    combiner.combine_chunks()
    
    # Print summary
    combiner.print_chunk_summary()
    
    # Add inter-chunk edges
    combiner.add_inter_chunk_edges()
    
    # Save combined graph
    combiner.save_combined_graph()
    
    # Print completion message
    print("\n" + "="*80)
    print("COMPLETION")
    print("="*80)
    print("✓ Combined DOT file created successfully!")
    print(f"✓ Output file: {output_file}")
    print()
    layers = combiner._detect_layers()
    print("Inter-chunk edges added for all layers:")
    print(f"  • Layers: {sorted(layers)}")
    print(f"  • cache_k_l{{L}} (chunk N) → Kcur-{{L}} X*Y (chunk N+1)")
    print(f"  • cache_v_l{{L}} (chunk N) → Vcur-{{L}} X*Y (chunk N+1)")
    chunk_connections = " → ".join(str(c.chunk_idx) for c in combiner.chunks)
    print(f"  • Chunk connections: {chunk_connections}")
    print()
    print(f"Total nodes: {combiner.combined_graph.number_of_nodes()}")
    print(f"Total edges: {combiner.combined_graph.number_of_edges()}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

