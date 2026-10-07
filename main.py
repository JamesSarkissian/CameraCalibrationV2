import calibrate
import undistort
from pathlib import Path

while True:

    print("\n===== Camera Calibration Tool =====")
    print("1. Calibrate Camera")
    print("2. Undistort Images")
    print("3. Exit")

    choice = input("> ")

    if choice == "1":
        calibrate.calibrate_camera("images")

    elif choice == "2":
        name = input("Image name (including extension): ").strip()
        image_path = Path(__file__).resolve().parent / "testImages" / name

        if not name or Path(name).name != name:
            print("Please enter only the image filename, including its extension.")
        elif not image_path.is_file():
            print(f"Image not found: {image_path}")
        else:
            undistort.undistort_image(str(image_path))

    elif choice == "3":
        break

    else:
        print("Invalid option.")
