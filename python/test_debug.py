# python/test_debug.py

import sys
import traceback

def test_basic_imports():
    print("Testing basic MLIR imports...")
    try:
        import mlir.ir
        print("✓ mlir.ir imported successfully")
    except Exception as e:
        print(f"✗ Failed to import mlir.ir: {e}")
        return False
    
    try:
        import mlir.dialects
        print("✓ mlir.dialects imported successfully")
    except Exception as e:
        print(f"✗ Failed to import mlir.dialects: {e}")
        return False
    
    try:
        import mlir.dialects.func
        print("✓ mlir.dialects.func imported successfully")
    except Exception as e:
        print(f"✗ Failed to import mlir.dialects.func: {e}")
        return False
    
    return True

def test_sar_dialect_import():
    print("\nTesting SAR dialect imports...")
    try:
        import mlir.dialects.sar
        from mlir.ir import Context
        print("✓ mlir.dialects.sar imported successfully")
        
        # Test registration with explicit context
        # Create a fresh context to avoid conflicts
        ctx = Context()
        mlir.dialects.sar.register_dialect(ctx)
        print("✓ SAR dialect registered successfully")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import SAR dialect: {e}")
        traceback.print_exc()
        return False

def test_sar_types():
    print("\nTesting SAR types...")
    try:
        from mlir.dialects.sar import TensorType, register_dialect
        from mlir.ir import Context, F32Type
        
        # Use a fresh context and register SAR dialect on it
        ctx = Context()
        register_dialect(ctx, load=True)
        with ctx:
            f32 = F32Type.get()
            tensor_type = TensorType.get([8, 4, 2], f32, context=ctx)
            print(f"✓ Created SAR TensorType: {tensor_type}")
            return True
    except Exception as e:
        print(f"✗ Failed to create SAR types: {e}")
        traceback.print_exc()
        return False

def test_frontend():
    print("\nTesting frontend...")
    try:
        from mlir.dialects.sar import sar_func, float32, fft_ndim, fft_dimx
        print("✓ Frontend imports successful")
        
        @sar_func
        def simple_forward(arg0: float32[8, 4, 2]) -> float32[8, 4, 2]:
            var0 = fft_ndim(arg0)
            return var0
        
        module = simple_forward()
        module_str = str(module)
        print(f"✓ Generated MLIR:\n{module_str}")
        
        # Basic checks
        assert "sar.fft_ndim" in module_str, "Missing sar.fft_ndim in generated MLIR"
        print("✓ All frontend tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Frontend test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("SAR Dialect Debug Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_basic_imports()
    all_passed &= test_sar_dialect_import()
    all_passed &= test_sar_types()
    all_passed &= test_frontend()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 