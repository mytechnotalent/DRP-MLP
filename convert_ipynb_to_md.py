#!/usr/bin/env python3

"""
Script to convert Jupyter Notebook (.ipynb) files to Markdown using jupyter nbconvert.
Requires Jupyter (nbconvert) to be installed in the environment.

Usage:
    python convert_ipynb_to_md.py <input_ipynb> [output_md]

If `output_md` is not provided, the Markdown file will be created next to the input
notebook with the same base name and a .md extension. Any accompanying resources
images folder produced by nbconvert (e.g. `notebook_files/`) will be moved/renamed
to match the output basename if an explicit output path is provided.
"""

import subprocess
import sys
import os
import shutil
import re


def _try_import_nbconvert():
    """
    Attempt to import the nbconvert module.

    Returns:
        bool: True if nbconvert can be imported, False otherwise.
    """
    try:
        import nbconvert

        return True
    except Exception:
        return False


def _upgrade_pip():
    """
    Upgrade pip to the latest version using subprocess.

    Returns:
        bool: True if pip upgrade succeeds, False otherwise.
    """
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Automatic pip install failed: {e}")
        return False


def _install_nbconvert_packages():
    """
    Install nbconvert and jupyter packages using pip.

    Returns:
        bool: True if installation succeeds, False otherwise.
    """
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "nbconvert", "jupyter"]
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _reimport_nbconvert():
    """
    Re-import nbconvert after installation by invalidating import caches.

    Returns:
        bool: True if nbconvert can be imported after cache invalidation, False otherwise.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        import nbconvert

        return True
    except Exception:
        return False


def _ensure_nbconvert_installed():
    """
    Ensure that nbconvert/jupyter is available in the current Python environment.

    Returns:
        bool: True if nbconvert is available or successfully installed, False otherwise.
    """
    if _try_import_nbconvert():
        return True
    print("`nbconvert` not found — attempting to install via pip...")
    if not _upgrade_pip() or not _install_nbconvert_packages():
        return False
    return _reimport_nbconvert()


def _validate_input_file(input_file):
    """
    Validate that the input file exists.

    Args:
        input_file (str): Path to the input .ipynb file.

    Returns:
        bool: True if file exists, False otherwise.
    """
    if os.path.exists(input_file):
        return True
    print(f"Error: Input file '{input_file}' does not exist.")
    return False


def _parse_input_path(input_file):
    """
    Parse input file to extract absolute path, directory, and base name.

    Args:
        input_file (str): Path to the input .ipynb file.

    Returns:
        tuple: (input_file_abs, input_dir, input_base)
    """
    input_file_abs = os.path.abspath(input_file)
    input_dir = os.path.dirname(input_file_abs) or "."
    input_base = os.path.splitext(os.path.basename(input_file_abs))[0]
    return input_file_abs, input_dir, input_base


def _build_custom_output_path(output_file_abs, out_base):
    """
    Ensure output path has .md extension.

    Args:
        output_file_abs (str): Absolute path to output file.
        out_base (str): Base name of output file.

    Returns:
        str: Output file path with .md extension.
    """
    if os.path.splitext(output_file_abs)[1].lower() != ".md":
        out_dir = os.path.dirname(output_file_abs) or "."
        output_file_abs = os.path.join(out_dir, out_base + ".md")
    return output_file_abs


def _resolve_output_path(input_dir, input_base, output_file):
    """
    Resolve output file path and base name from optional output argument.

    Args:
        input_dir (str): Directory of input file.
        input_base (str): Base name of input file.
        output_file (str): Optional output file path.

    Returns:
        tuple: (output_file_abs, out_base)
    """
    if output_file:
        output_file_abs = os.path.abspath(output_file)
        out_base = os.path.splitext(os.path.basename(output_file_abs))[0]
        output_file_abs = _build_custom_output_path(output_file_abs, out_base)
    else:
        output_file_abs = os.path.join(input_dir, input_base + ".md")
        out_base = input_base
    return output_file_abs, out_base


def _compute_paths(input_file, output_file):
    """
    Compute absolute paths and base names for input and output files.

    Args:
        input_file (str): Path to the input .ipynb file.
        output_file (str): Optional path to the output .md file.

    Returns:
        tuple: (input_file_abs, input_dir, input_base, output_file_abs, out_base)
    """
    input_file_abs, input_dir, input_base = _parse_input_path(input_file)
    output_file_abs, out_base = _resolve_output_path(input_dir, input_base, output_file)
    return input_file_abs, input_dir, input_base, output_file_abs, out_base


def _get_nbconvert_base_args():
    """
    Get base arguments for jupyter nbconvert command.

    Returns:
        list: Base arguments list for nbconvert.
    """
    return [sys.executable, "-m", "nbconvert", "--to", "markdown"]


def _build_nbconvert_command(input_file_abs, input_base):
    """
    Build the jupyter nbconvert command as a list.

    Args:
        input_file_abs (str): Absolute path to input .ipynb file.
        input_base (str): Base name of the input file without extension.

    Returns:
        list: Command list for subprocess execution.
    """
    cmd = _get_nbconvert_base_args()
    cmd.extend([os.path.basename(input_file_abs), "--output", input_base])
    return cmd


def _execute_nbconvert(input_file_abs, input_dir, input_base):
    """
    Execute jupyter nbconvert to convert .ipynb to markdown.

    Args:
        input_file_abs (str): Absolute path to input .ipynb file.
        input_dir (str): Directory containing the input file.
        input_base (str): Base name of the input file without extension.

    Returns:
        subprocess.CompletedProcess: Result of the nbconvert subprocess execution.
    """
    cmd = _build_nbconvert_command(input_file_abs, input_base)
    result = subprocess.run(cmd, cwd=input_dir, capture_output=True, text=True)
    return result


def _read_markdown_file(generated_md):
    """
    Read the contents of the generated markdown file.

    Args:
        generated_md (str): Path to the generated markdown file.

    Returns:
        str: Contents of the markdown file, or empty string if file doesn't exist.
    """
    if not os.path.exists(generated_md):
        return ""
    with open(generated_md, "r", encoding="utf-8") as fh:
        return fh.read()


def _remove_scoped_styles(md_text):
    """
    Remove scoped style blocks from markdown text.

    Args:
        md_text (str): Markdown text content.

    Returns:
        str: Markdown text with scoped style blocks removed.
    """
    return re.sub(
        r"<style[^>]*\bscoped\b[^>]*>.*?</style>\s*", "", md_text, flags=re.S | re.I
    )


def _replace_resource_paths(md_text, input_base, out_base):
    """
    Replace resource folder paths in markdown text.

    Args:
        md_text (str): Markdown text content.
        input_base (str): Base name of input file.
        out_base (str): Base name of output file.

    Returns:
        str: Markdown text with updated resource paths.
    """
    src_folder = input_base + "_files"
    dst_folder = out_base + "_files"
    if src_folder != dst_folder:
        md_text = md_text.replace(src_folder + "/", dst_folder + "/")
        md_text = md_text.replace("./" + src_folder + "/", dst_folder + "/")
    return md_text


def _write_markdown_file(generated_md, md_text):
    """
    Write updated markdown text to file.

    Args:
        generated_md (str): Path to the markdown file.
        md_text (str): Markdown text content to write.

    Returns:
        None
    """
    with open(generated_md, "w", encoding="utf-8") as fh:
        fh.write(md_text)


def _fix_resource_links(generated_md, input_base, out_base):
    """
    Fix resource links in the generated markdown file.

    Args:
        generated_md (str): Path to the generated markdown file.
        input_base (str): Base name of input file.
        out_base (str): Base name of output file.

    Returns:
        None
    """
    try:
        md_text = _read_markdown_file(generated_md)
        md_text = _remove_scoped_styles(md_text)
        md_text = _replace_resource_paths(md_text, input_base, out_base)
        _write_markdown_file(generated_md, md_text)
    except Exception as e:
        print(f"Warning: failed to rewrite resource links: {e}")


def _move_markdown_file(generated_md, output_file_abs):
    """
    Move the generated markdown file to the desired output location.

    Args:
        generated_md (str): Path to the generated markdown file.
        output_file_abs (str): Desired output path for the markdown file.

    Returns:
        None
    """
    if os.path.abspath(generated_md) != os.path.abspath(output_file_abs):
        os.makedirs(os.path.dirname(output_file_abs), exist_ok=True)
        shutil.move(generated_md, output_file_abs)


def _move_resources_folder(generated_resources, output_file_abs, out_base):
    """
    Move the generated resources folder to the desired location.

    Args:
        generated_resources (str): Path to the generated resources folder.
        output_file_abs (str): Output file path (used to determine destination directory).
        out_base (str): Base name for the output resources folder.

    Returns:
        None
    """
    if not os.path.isdir(generated_resources):
        return
    desired_resources = os.path.join(
        os.path.dirname(output_file_abs), out_base + "_files"
    )
    if os.path.isdir(desired_resources):
        shutil.rmtree(desired_resources)
    shutil.move(generated_resources, desired_resources)


def _validate_conversion_requirements(input_file):
    """
    Validate that input file exists and nbconvert is available.

    Args:
        input_file (str): Path to the input .ipynb file.

    Returns:
        bool: True if all requirements are met, False otherwise.
    """
    if not _validate_input_file(input_file):
        return False
    if not _ensure_nbconvert_installed():
        print(
            "Error: 'nbconvert' (jupyter) is not available and automatic install failed."
        )
        print("Install manually with: pip install nbconvert jupyter")
        return False
    return True


def _execute_conversion_subprocess(input_file_abs, input_dir, input_base):
    """
    Execute nbconvert subprocess and validate the result.

    Args:
        input_file_abs (str): Absolute path to input .ipynb file.
        input_dir (str): Directory containing the input file.
        input_base (str): Base name of the input file.

    Returns:
        bool: True if conversion succeeds, False otherwise.
    """
    result = _execute_nbconvert(input_file_abs, input_dir, input_base)
    if result.returncode != 0:
        print("Error: jupyter nbconvert failed:")
        print(result.stderr)
        return False
    return True


def _process_generated_files(
    input_dir, input_base, output_file_abs, out_base, input_file
):
    """
    Process generated markdown and resource files.

    Args:
        input_dir (str): Directory containing generated files.
        input_base (str): Base name of input file.
        output_file_abs (str): Desired output file path.
        out_base (str): Base name for output.
        input_file (str): Original input file path for success message.

    Returns:
        None
    """
    generated_md = os.path.join(input_dir, input_base + ".md")
    generated_resources = os.path.join(input_dir, input_base + "_files")
    _fix_resource_links(generated_md, input_base, out_base)
    _move_markdown_file(generated_md, output_file_abs)
    _move_resources_folder(generated_resources, output_file_abs, out_base)
    print(f"Successfully converted '{input_file}' → '{output_file_abs}'")


def _handle_conversion_exception(e):
    """
    Handle and report conversion exceptions with appropriate error messages.

    Args:
        e (Exception): The exception that was raised.

    Returns:
        None
    """
    if isinstance(e, FileNotFoundError):
        print("Error: 'jupyter' (nbconvert) is not installed or not on PATH.")
        print("Install with: pip install nbconvert jupyter")
    else:
        print(f"An unexpected error occurred: {e}")


def _run_conversion_workflow(input_file, output_file):
    """
    Execute the conversion workflow with error handling.

    Args:
        input_file (str): Path to the input .ipynb file.
        output_file (str): Optional output file path.

    Returns:
        bool: True on success, False on failure.
    """
    try:
        paths = _compute_paths(input_file, output_file)
        if not _execute_conversion_subprocess(paths[0], paths[1], paths[2]):
            return False
        _process_generated_files(paths[1], paths[2], paths[3], paths[4], input_file)
        return True
    except Exception as e:
        _handle_conversion_exception(e)
        return False


def convert_ipynb_to_md(input_file, output_file=None):
    """
    Convert an .ipynb file to Markdown using jupyter nbconvert.

    Args:
        input_file (str): Path to the input .ipynb file.
        output_file (str, optional): Desired output .md file path.

    Returns:
        bool: True on success, False on failure.
    """
    if not _validate_conversion_requirements(input_file):
        return False
    return _run_conversion_workflow(input_file, output_file)


def main():
    """
    Main entry point for the script.

    Parses command-line arguments and invokes the conversion function.

    Returns:
        None (exits with code 0 on success, 1 on failure).
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_ipynb_to_md.py <input_ipynb> [output_md]")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    ok = convert_ipynb_to_md(input_file, output_file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
