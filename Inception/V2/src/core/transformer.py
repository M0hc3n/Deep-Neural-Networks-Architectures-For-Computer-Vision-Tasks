import os
import nbformat as nbf


def convert_to_notebook(project_dir, main_file, folder_order, last_files=None):
    nb = nbf.v4.new_notebook()

    def add_code_cell(code):
        code_cell = nbf.v4.new_code_cell(code)
        nb.cells.append(code_cell)

    for folder in folder_order:
        folder_path = os.path.join(project_dir, folder)
        if os.path.exists(folder_path):
            files_in_folder = []
            last_file = None

            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".py") and file != main_file:
                        if last_files and file in last_files:
                            last_file = file
                        else:
                            files_in_folder.append(file)

            for file in files_in_folder:
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as f:
                    code = f.read()

                add_code_cell(code)

            if last_file:
                last_file_path = os.path.join(folder_path, last_file)
                with open(last_file_path, "r") as f:
                    last_code = f.read()

                add_code_cell(last_code)

    main_file_path = os.path.join(project_dir, main_file)
    if os.path.exists(main_file_path):
        with open(main_file_path, "r") as f:
            main_code = f.read()

        add_code_cell(main_code)

    with open("./src/inception_v2.ipynb", "w") as f:
        nbf.write(nb, f)


# Usage Example
folder_order = ["preparation", "modeling"]
last_files = {
    "preparation": "extract.py",
    "modeling": "train.py",
}
convert_to_notebook("./src", "main.py", folder_order, last_files)
