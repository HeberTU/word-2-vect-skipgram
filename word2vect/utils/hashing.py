# -*- coding: utf-8 -*-
"""Hashing utilities.

Created on: 2/8/22
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import hashlib
from typing import Any


def hash_object_repr(obj: Any) -> str:
    """Create a hash representation of an object.

    Args:
        obj: Any python object.

    Returns:
        hashed_obj: hashed object.
    """
    return hashlib.sha512(repr(obj).encode()).hexdigest()
