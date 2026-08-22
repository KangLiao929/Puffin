import os
import shutil

def copy_images_with_folder_prefix(src_folder: str, dest_folder: str, extensions=None):
    """
    Copy all images from src_folder into dest_folder, 
    renaming each to foldername_originalfilename.

    :param src_folder: Path to source folder containing images.
    :param dest_folder: Path to destination folder.
    :param extensions: Optional set of lowercase extensions to include (e.g. {'jpg','png'}).
                       If None, defaults to common image formats.
    """
    if extensions is None:
        extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp'}

    # Ensure destination exists
    os.makedirs(dest_folder, exist_ok=True)

    # Get the base name of the source folder
    folder_name = os.path.basename(os.path.normpath(src_folder))

    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        if not os.path.isfile(src_path):
            continue

        # Check extension
        base, ext = os.path.splitext(fname)
        if ext.lower().lstrip('.') not in extensions:
            continue

        # Build new filename and copy
        new_name = f"{folder_name}_{fname}"
        dest_path = os.path.join(dest_folder, new_name)
        shutil.copy2(src_path, dest_path)

    print(f"Copied images from '{src_folder}' to '{dest_folder}' with prefix '{folder_name}_'.")


if __name__ == "__main__":
    source_directory = r"/mnt/sfs-common/kliao/Code/Puffins/Show-o/.scripts/GeoCalib-main/HDR2"
    destination_directory = r"/mnt/sfs-common/kliao/Dataset/Puffins/panorama-source/train_dataset/HDR/"
    copy_images_with_folder_prefix(source_directory, destination_directory)
