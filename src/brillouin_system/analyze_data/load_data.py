import pickle


def load_measurements(pickle_path):
    try:
        with open(pickle_path, "rb") as f:
            measurements = pickle.load(f)
        print(f"[✓] Loaded {len(measurements)} measurements from {pickle_path}")
        return measurements
    except Exception as e:
        print(f"[✗] Failed to load measurements: {e}")
        return []