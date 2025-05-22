from brillouinDAQ.devices.zaber_linear import ZaberLinearController


def main():
    print("Connecting to Zaber device...")
    try:
        zaber = ZaberLinearController(port="COM5")  # Replace with your actual port

        print("Homing device...")
        zaber.home()

        pos = zaber.get_position('x')
        print(f"Initial position: {pos:.2f} µm")

        print("Moving +500 µm...")
        zaber.move_rel('x',500)
        pos = zaber.get_position('x')
        print(f"Position after move_rel(+500): {pos:.2f} µm")

        print("Moving to absolute position 1000 µm...")
        zaber.move_abs('x',1000)
        pos = zaber.get_position('x')
        print(f"Position after move_abs(1000): {pos:.2f} µm")

        print("Setting speed to 10 mm/s and moving +250 µm...")
        zaber.set_speed('x',10)
        zaber.move_rel('x',250)
        pos = zaber.get_position('x')
        print(f"Position after move_rel(+250): {pos:.2f} µm")

        print("Test complete.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'zaber' in locals():
            zaber.close()

if __name__ == "__main__":
    main()
