#!/usr/bin/env python3

#########
#IMPORTS#
#########

import pandas as pd 
import numpy as np 
from Bio import SeqIO 
import os
import peptides 
from argparse import ArgumentParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint


def calculate_molecular_weight(peptide):
    analyzed_peptide = ProteinAnalysis(peptide)
    return analyzed_peptide.molecular_weight()

def calculate_aromaticity(peptide):
    analyzed_peptide = ProteinAnalysis(peptide)
    return analyzed_peptide.aromaticity()

def calculate_instability_index(peptide):
    analyzed_peptide = ProteinAnalysis(peptide)
    return analyzed_peptide.instability_index()

def calculate_helix(peptide):
    analyzed_peptide = ProteinAnalysis(peptide)
    return analyzed_peptide.secondary_structure_fraction()[0]

def calculate_cys_red(peptide):
    analyzed_peptide = ProteinAnalysis(peptide)
    return analyzed_peptide.molar_extinction_coefficient()[0]

def calculate_IP(peptide):
    peptideIP = peptides.Peptide(peptide)
    analyzed_peptide_IP  = peptideIP.isoelectric_point()
   # analyzed_peptide = IsoelectricPoint(peptide)
    return analyzed_peptide_IP


def calculate_and_save_properties(df, peptide_column_name, weight_column_name, aromaticity_column_name, instability_column_name, helix_column_name,cys_red_cloumn_name,IP_column_name):
    df[weight_column_name] = df[peptide_column_name].apply(calculate_molecular_weight)
    df[aromaticity_column_name] = df[peptide_column_name].apply(calculate_aromaticity)
    df[instability_column_name] = df[peptide_column_name].apply(calculate_instability_index)
    df[helix_column_name] = df[peptide_column_name].apply(calculate_helix)
    df[cys_red_cloumn_name] = df[peptide_column_name].apply(calculate_cys_red)
    df[IP_column_name] = df[peptide_column_name].apply(calculate_IP)
    return(df)

