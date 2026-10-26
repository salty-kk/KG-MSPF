# TCGA-BRCA input

Run `fetch_tcga.py` from the repository root with the pinned revision shown in the main README. It downloads public clinical and expression tables from https://github.com/cBioPortal/datahub and prepares `tcga_brca_harmonized.csv` with a source-version sidecar.

Primary study: `brca_tcga_pan_can_atlas_2018`; the pipeline also obtains clinical fields from `brca_tcga`. The evaluated shared-feature cohort has 1,069 patients and 151 deaths. Mutation files are not downloaded; unavailable calls are not converted to wild type. The original source-file versions and fingerprints are in `../data_sources.json`.

The downloaded and harmonized patient-level tables remain local and are ignored by Git. The software license does not override the data providers' terms.
