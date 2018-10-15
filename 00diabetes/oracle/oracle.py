#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:56:37 2018

@author: pan
"""
import cx_Oracle
import pandas as pd
import platform

if __name__ == '__main__':
    db = cx_Oracle.connect('system', 'oracle', 'localhost:1521/APEX_040200')
    pass