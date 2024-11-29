def test_parse_counter():
    test_cases = [
        # Valid cases
        ("checkpoint_iter_42.pth", 42),
        ("checkpoint_5.pth", 5),
        ("iter_100.pth", 100),
        # Invalid cases
        ("checkpoint.pth", 0),       # No numeric suffix
        ("checkpoint_iter_.pth", 0), # No number after "iter_"
        ("checkpoint_iter_abc.pth", 0), # Non-numeric suffix
        ("checkpoint_iter.pth", 0),  # No suffix after "iter"
        ("checkpoint_.pth", 0),      # Underscore but no number
        ("checkpoint.pth", 0),       # Just "checkpoint"
        # Edge cases
        ("checkpoint_iter_0.pth", 0), # Numeric zero
        ("checkpoint___.pth", 0),     # Just underscores
        ("241001_checkpoint___.pth", 0),     # sg else
    ]

    for filename, expected in test_cases:
        filename_parts = filename.split('/')[-1].split('.')[0].split('_')
        if len(filename_parts) > 1 and filename_parts[-1].isdigit():
            counter = int(filename_parts[-1])
        else:
            counter = 0

        assert counter == expected, f"Failed for filename: {filename}, got {counter}, expected {expected}"

    print("All test cases passed!")

# Run the test
test_parse_counter()
