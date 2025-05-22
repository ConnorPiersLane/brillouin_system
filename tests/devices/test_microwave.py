from brillouinDAQ.devices.microwave_device import Microwave  # Replace with actual import path

def test_microwave_real():
    print("=== Microwave Device Real Hardware Test ===")

    try:
        mw = Microwave(port="COM3")  # Change COM3 if needed
        print("[Test] Microwave initialized successfully.")
    except RuntimeError as e:
        raise RuntimeError(f"[Test] Failed to initialize Microwave: {e}")

    # Test frequency setting and getting
    print("[Test] Setting frequency to 5.8 GHz...")
    mw.set_frequency(5.8)
    freq = mw.get_frequency()
    assert abs(freq - 5.8) < 0.01, f"[Test] Frequency mismatch: expected 5.8 GHz, got {freq} GHz"

    # Test power setting and getting
    print("[Test] Setting power to +2.5 dBm...")
    mw.set_power(2.5)
    power = mw.get_power()
    assert abs(power - 2.5) < 0.2, f"[Test] Power mismatch: expected 2.5 dBm, got {power} dBm"

    # Test output enable/disable
    print("[Test] Enabling output...")
    mw.enable_output(True)
    assert mw.is_output_enabled() is True, "[Test] Output should be ON but isn't"

    print("[Test] Disabling output...")
    mw.enable_output(False)
    assert mw.is_output_enabled() is False, "[Test] Output should be OFF but isn't"

    # Shutdown
    print("[Test] Shutting down Microwave...")
    mw.shutdown()

    print("\nAll Microwave hardware tests PASSED.")

if __name__ == "__main__":
    test_microwave_real()
