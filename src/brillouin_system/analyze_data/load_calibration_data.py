import pickle
from brillouinDAQ.my_dataclasses.calibration_data import CalibrationData

def load_calibration(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, CalibrationData):
            print("[✓] Loaded Calibration Data")
            print(f" - Spectral Dispersion (SD): {data.sd:.6f} GHz/px")
            print(f" - Free Spectral Range (FSR): {data.fsr:.3f} GHz")
            print(f" - Number of calibration points: {len(data.fitted_spectras)}")
            return data
        else:
            raise TypeError("Loaded object is not a CalibrationData instance")

    except Exception as e:
        print(f"[✗] Failed to load calibration data: {e}")
        return None

if __name__ == "__main__":
    path = r"C:\Users\cplan\Documents\repos\brillouinDAQ\src\brillouinDAQ\apps\calibration.pkl"
    data = load_calibration(path)
    print(1)

