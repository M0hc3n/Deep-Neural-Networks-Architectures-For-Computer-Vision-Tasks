import os
import nbformat as nbf

def convert_to_notebook(project_dir, main_file, folder_order):
    nb = nbf.v4.new_notebook()

    def add_code_cell(code):
        code_cell = nbf.v4.new_code_cell(code)
        nb.cells.append(code_cell)

    for folder in folder_order:
        folder_path = os.path.join(project_dir, folder)
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".py") and file != main_file:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            code = f.read()

                        add_code_cell(code)
    
    main_file_path = os.path.join(project_dir, main_file)
    if os.path.exists(main_file_path):
        with open(main_file_path, "r") as f:
            main_code = f.read()

        add_code_cell(main_code)
    
    with open("./src/project_notebook.ipynb", "w") as f:
        nbf.write(nb, f)

folder_order = ["core", "preparation", "modeling"] 
convert_to_notebook("./src", "main.py", folder_order)
