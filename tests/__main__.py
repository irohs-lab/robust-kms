import importlib
import pkgutil
import unittest
import tests


def main():
  suite = unittest.TestSuite()
  for entry in pkgutil.iter_modules(tests.__path__):
    if not entry.name.startswith("test_"):
      continue
    module = importlib.import_module("tests." + entry.name)
    for name, function in vars(module).items():
      if name.startswith("test_") and callable(function):
        suite.addTest(unittest.FunctionTestCase(function))
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
  main()
