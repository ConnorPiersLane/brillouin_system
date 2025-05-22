from brillouinDAQ.devices.shutter_device import (
    Shutter,
    ShutterDummy,
    ShutterManager,
    ShutterManagerDummy
)

def test_individual_shutter_real():
    print("Testing individual real shutter initialization...")
    s = Shutter(384)
    s.set_state(True)
    assert s.get_state() is True, "Failed to open shutter"
    s.set_state(False)
    assert s.get_state() is False, "Failed to close shutter"
    s.shutdown()

def test_individual_shutter_dummy():
    print("Testing individual dummy shutter initialization...")
    s = ShutterDummy(384)
    s.open()
    assert s.get_state() is True, "Failed to open dummy shutter"
    s.close()
    assert s.get_state() is False, "Failed to close dummy shutter"
    s.shutdown()

def test_shutter_manager_dummy(sample="human_interface"):
    print(f"Testing ShutterManagerDummy with sample='{sample}'...")
    sc = ShutterManagerDummy(sample)

    print("  -> Setting to OBJECTIVE state")
    sc.change_to_objective()
    assert sc.objective.get_state() is True
    assert sc.reference.get_state() is False

    print("  -> Setting to REFERENCE state")
    sc.change_to_reference()
    assert sc.objective.get_state() is False
    assert sc.reference.get_state() is True

    print("  -> Closing all shutters")
    sc.close_all()
    assert sc.objective.get_state() is False
    assert sc.reference.get_state() is False
    assert sc.sample.get_state() is False

    sc.shutdown_all()

# Optional: create a real hardware test if hardware is available
def test_shutter_manager_real(sample="human_interface"):
    print(f"Testing ShutterManager with sample='{sample}'...")
    sc = ShutterManager(sample)

    print("  -> Setting to OBJECTIVE state")
    sc.change_to_objective()
    assert sc.objective.get_state() is True
    assert sc.reference.get_state() is False

    print("  -> Setting to REFERENCE state")
    sc.change_to_reference()
    assert sc.objective.get_state() is False
    assert sc.reference.get_state() is True

    print("  -> Closing all shutters")
    sc.close_all()
    assert sc.objective.get_state() is False
    assert sc.reference.get_state() is False
    assert sc.sample.get_state() is False

    sc.shutdown_all()

if __name__ == "__main__":
    test_individual_shutter_dummy()
    print("\n---\n")
    # test_shutter_manager_dummy("human_interface")

    # Uncomment if real hardware is connected
    print("\n---\n")
    test_individual_shutter_real()
    print("\n---\n")
    test_shutter_manager_real("human_interface")

    print("\nAll shutter tests passed successfully.")
