import sys
import pickle
from mpi4py import MPI
from monty.serialization import loadfn

from HiPRGen.reaction_filter import (
    dispatcher,
    worker,
    DISPATCHER_RANK
)
import dgl
from HiPRGen.lmdb_dataset import load_dgl_graph_from_serialized
# python run_network_generation.py mol_entries_pickle_file dispatcher_payload.json worker_payload.json


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mol_entries_pickle_file = sys.argv[1]
dispatcher_payload_json = sys.argv[2]
worker_payload_json = sys.argv[3]
dgl_molecules_dict_pickle_file = sys.argv[4]
grapher_features_dict_pickle_file = sys.argv[5]
#wx,add reaction lmdb_path
reaction_lmdb_path = sys.argv[6]

print(rank, "loading mol_entries")
with open(mol_entries_pickle_file, 'rb') as f:
    mol_entries = pickle.load(f)

print(rank, "loading dgl_molecules_dict_pickle_file")
with open(dgl_molecules_dict_pickle_file, 'rb') as f:
    dgl_molecules_dict = pickle.load(f)

print(rank, "executing load_dgl_graph_from_serialized")
for graph_i in dgl_molecules_dict:
    dgl_molecules_dict[graph_i] = load_dgl_graph_from_serialized(dgl_molecules_dict[graph_i])

print(rank, "loading grapher_features_dict_pickle_file")
with open(grapher_features_dict_pickle_file, 'rb') as f:
    grapher_features_dict = pickle.load(f)

if rank == DISPATCHER_RANK:
    dispatcher_payload = loadfn(dispatcher_payload_json)
    dispatcher(mol_entries, 
               dispatcher_payload)
    
    #move to worker level
    # dispatcher(mol_entries,
    #            dgl_molecules_dict_pickle_file,
    #            grapher_features_dict_pickle_file,
    #            dispatcher_payload,
    #            #wx,
    #            reaction_lmdb_path
    #            )
else:
    worker_payload = loadfn(worker_payload_json)
    worker(mol_entries,
           worker_payload,
           dgl_molecules_dict,
           grapher_features_dict,
           reaction_lmdb_path
           )
