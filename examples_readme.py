import os

EXAMPLES_DIR = "examples"
README_FILE = os.path.join(EXAMPLES_DIR, "README.md")

HEADER_TEMPLATE = """# Examples

This directory contains a collection of examples that demonstrate the usage of various modules and functionalities in this project. Each subfolder corresponds to a specific module and includes example scripts to help you understand how to use that module.

## Directory Structure

The examples are organized as follows:

```
{directory_structure}
```

## How to Use

1. Navigate to the module folder of interest, e.g., `examples/module1/`.
2. Open the `README.md` in that folder to get detailed information about the examples.
3. Run the scripts directly using:
   ```bash
   python example1.py
   ```

## Modules and Examples
"""

MODULE_TEMPLATE = """
### {module_name}

#### Description
{description}

{examples}
"""

EXAMPLE_TEMPLATE = """
- **{file_name}**: {description}
```python
{code_snippet}
  ```
"""


def generate_directory_structure(base_dir):
    """Generates the directory structure in a tree-like format."""
    structure = []
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, "").count(os.sep)
        indent = "    " * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = "    " * (level + 1)
        for file in sorted(files):
            if file.endswith(".py"):
                structure.append(f"{sub_indent}{file}")
    return "\n".join(structure)


def extract_code_snippet(file_path):
    """Extracts the full content of a Python file."""
    try:
        with open(file_path, "r") as file:
            content = file.read()
            return content.strip()
    except Exception as e:
        return f"# Error reading file: {e}"


def generate_module_section(module_path, module_name):
    """Generates the section for a specific module, including its examples."""
    examples = []
    for file in sorted(os.listdir(module_path)):
        if file.endswith(".py"):
            file_path = os.path.join(module_path, file)
            code_snippet = extract_code_snippet(file_path)
            examples.append(
                EXAMPLE_TEMPLATE.format(
                    file_name=file,
                    description="Example demonstrating functionality.",
                    code_snippet=code_snippet,
                )
            )
    return MODULE_TEMPLATE.format(
        module_name=module_name,
        description="This module demonstrates specific functionalities.",
        examples="\n".join(examples),
    )


def generate_readme(base_dir, readme_file):
    """Generates the README.md file for the examples directory."""
    directory_structure = generate_directory_structure(base_dir)
    readme_content = HEADER_TEMPLATE.format(directory_structure=directory_structure)

    # Add module-specific sections
    for module in sorted(os.listdir(base_dir)):
        module_path = os.path.join(base_dir, module)
        if os.path.isdir(module_path):
            readme_content += generate_module_section(module_path, module) + "\n"

    # Write to README.md
    with open(readme_file, "w") as readme:
        readme.write(readme_content)


if __name__ == "__main__":
    generate_readme(EXAMPLES_DIR, README_FILE)
