"""Original synthetic shader inputs, independent of any application format."""
def fixture(path, pixels=1):
    path.write_text(str(pixels))
