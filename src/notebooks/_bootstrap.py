# Databricks notebook source
# Shared bootstrap, pulled in with %run: puts the `chiro` package (../chiro) on sys.path.
import os
import sys

_SRC = os.path.abspath("..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
