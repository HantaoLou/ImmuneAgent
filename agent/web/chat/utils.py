from typing import Any

import yaml


def state_to_markdown(state: dict[str, Any]) -> str:
    """
    Convert a dictionary to markdown format.

    Args:
        state: Dictionary to convert to markdown

    Returns:
        Markdown string representation of the dictionary
    """

    def _convert_value(value: Any, level: int = 0) -> str:
        """Recursively convert values to markdown."""
        if isinstance(value, dict):
            # Convert nested dictionary to sections
            sections = []
            for key, val in value.items():
                # Always create a header for dictionary keys
                sections.append(f"{'#' * (level + 2)} {key}")
                if isinstance(val, (dict, list)):
                    # For complex values, process them recursively
                    sections.append(_convert_value(val, level + 1))
                else:
                    # For simple values, display them below the header
                    sections.append(str(val))
            return "\n\n".join(sections)

        elif isinstance(value, list):
            # Convert list to markdown list
            if not value:
                return "* (empty list)"

            list_items = []
            for item in value:
                if isinstance(item, dict):
                    # For dictionary items in lists, convert them to sections
                    item_md = _convert_value(item, level + 1)
                    list_items.append(f"* {item_md}")
                else:
                    list_items.append(f"* {item}")
            return "\n".join(list_items)

        else:
            # Convert simple values to string
            return str(value)

    # Process the top-level dictionary
    markdown_parts = []
    for key, value in state.items():
        # Always create a header for every key
        markdown_parts.append(f"# {key}")

        if isinstance(value, (dict, list)):
            # For complex values, process them recursively
            markdown_parts.append(_convert_value(value, 1))
        else:
            # For simple values, display them below the header
            markdown_parts.append(str(value))

    return "\n\n".join(markdown_parts)


if __name__ == "__main__":
    state = {
        "name": {
            "first": "John",
            "last": "Doe",
            "sub": {"name": "Jane", "age": 25, "city": "Los Angeles"},
        },
        "array": [1, 2, 3],
        "age": 30,
        "city": "New York",
    }
    print(state_to_markdown(state))
