"""Raw capture -> MANO layer: the drivers that run the dataset readers (paper Sec. 3.2, first step).

Two drivers, chosen by the reader registry (``dexengine.data.readers.READERS``):

- ``stage`` (``python -m dexengine.data.read_raw <dataset> read|intake ...``, module ``stage``): re-reads a dataset from
  its raw recordings into a staging folder (reader per recording, then decompose and contact), retargets the first
  hand there and ingests mano + that hand into the data layout, then runs the dataset's INTAKE derivations.
- ``hoi`` (``python -m dexengine.data.read_raw.hoi ...``): HO-Cap and HRDexDB (and DexCanvas) through their library
  readers, each sequence run through the whole retargeting chain (keypoints, decompose, contact, scene, IK) with a
  resume-safe status log.
"""
