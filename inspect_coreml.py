import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
import sys

def analyze_mlpackage(mlpackage_path):
    print(f"Loading {mlpackage_path}...")
    model = ct.models.MLModel(mlpackage_path)
    spec = model.get_spec()
    
    print("\n--- Inputs ---")
    for _input in spec.description.input:
        print(f"Name: {_input.name}")
        if _input.type.HasField('multiArrayType'):
            dims = _input.type.multiArrayType.shape
            print(f"  Shape: {dims}")
            
            # Check for allowed range constraints
            if _input.type.multiArrayType.HasField('shapeRange'):
                sr = _input.type.multiArrayType.shapeRange
                print("  Range Constraints:")
                for i, r in enumerate(sr.sizeRanges):
                    print(f"    Dim {i}: [{r.lowerBound}, {r.upperBound}]")
    
    # ML Program analysis
    if spec.HasField('mlProgram'):
        print("\n--- ML Program Functions ---")
        for name, func in spec.mlProgram.functions.items():
            print(f"Function: {name}")
            
            # Search for 'add' operations that might fail broadcasting
            add_ops = []
            for block in func.block_specializations.values():
                for op in block.operations:
                    if op.type == "add":
                        add_ops.append(op)
            
            print(f"  Found {len(add_ops)} add operations.")
            
            # If there's an issue, print the first few add operations to see the inputs
            for i, op in enumerate(add_ops[:5]):
                print(f"    Add Op {i}:")
                for k, v in op.inputs.items():
                    print(f"      Input {k}: {v.arguments}")
                for v in op.outputs:
                    print(f"      Output: {v.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_mlpackage(sys.argv[1])
    else:
        print("Usage: python inspect_coreml.py <path_to_mlpackage>")
