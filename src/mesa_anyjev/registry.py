"""Frozen vocabularies that make questions learnable (DESIGN D1, D7).

Every string here is part of a ``Question`` option list and therefore of a ``Question.key``.
Editing one rotates that key: update ``questions.lock.json`` (``mesa-anyjev questions
--update-lock``), say how labels migrate, and expect a refit.

Ontology registry rule (D7): the ten ontologies allowed by neon-avu-eval ``prompt.md`` plus
every CURIE prefix with at least five valid AVUs in ``results/validated.json``
(RESEARCH.md: TAXRANK 7, GENEPIO 5; GO has 3 and is out).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

Aspect = str


@dataclass(frozen=True)
class OntologyEntry:
    id: str
    curie_prefix: str
    option_text: str
    aspects: frozenset[Aspect]


ASPECTS: Final[tuple[str, ...]] = (
    "taxon",
    "environment",
    "method",
    "measurement",
    "unit",
    "data_type",
    "location",
    "other",
)

ASPECT_OPTIONS: Final[tuple[str, ...]] = (
    "taxon: the organisms observed, sampled or identified",
    "environment: the biome, habitat or environmental material at a site",
    "method: the sampling protocol, device, assay or processing step",
    "measurement: a measured quantity, quality or attribute of a thing",
    "unit: the unit of measure of a measured quantity",
    "data_type: the kind of record, identifier, dataset or information artifact",
    "location: a named place, region, plot or geographic feature",
    "other: none of the above or not an annotatable column",
)

ONTOLOGY_REGISTRY: Final[tuple[OntologyEntry, ...]] = (
    OntologyEntry(
        "envo",
        "ENVO",
        "ENVO: environments, biomes, habitats and environmental materials",
        frozenset({"environment", "location", "measurement"}),
    ),
    OntologyEntry(
        "ncbitaxon",
        "NCBITaxon",
        "NCBITaxon: organisms and taxa at every rank",
        frozenset({"taxon"}),
    ),
    OntologyEntry(
        "pato",
        "PATO",
        "PATO: qualities and attributes of things (size, sex, age, colour)",
        frozenset({"measurement"}),
    ),
    OntologyEntry(
        "uo",
        "UO",
        "UO: units of measurement (meter, degree Celsius, percent)",
        frozenset({"unit"}),
    ),
    OntologyEntry(
        "obi",
        "OBI",
        "OBI: investigations, assays, protocols, devices and specimens",
        frozenset({"method", "measurement", "data_type"}),
    ),
    OntologyEntry(
        "iao",
        "IAO",
        "IAO: information artifacts, data items, identifiers and documents",
        frozenset({"data_type", "method"}),
    ),
    OntologyEntry(
        "pco",
        "PCO",
        "PCO: populations, communities and their collective properties",
        frozenset({"taxon", "measurement"}),
    ),
    OntologyEntry(
        "bco",
        "BCO",
        "BCO: biological collections, sampling and identification processes",
        frozenset({"method"}),
    ),
    OntologyEntry(
        "gaz",
        "GAZ",
        "GAZ: named geographic places and administrative regions",
        frozenset({"location", "environment"}),
    ),
    OntologyEntry(
        "ro",
        "RO",
        "RO: relations between entities (part of, located in, participates in)",
        frozenset({"other"}),
    ),
    OntologyEntry(
        "taxrank",
        "TAXRANK",
        "TAXRANK: taxonomic ranks (species, genus, family, order)",
        frozenset({"taxon", "data_type"}),
    ),
    OntologyEntry(
        "genepio",
        "GENEPIO",
        "GENEPIO: sample collection, identification and metadata fields",
        frozenset({"method", "data_type"}),
    ),
)

ONTOLOGY_OPTIONS: Final[tuple[str, ...]] = tuple(e.option_text for e in ONTOLOGY_REGISTRY)

VALUE_KINDS: Final[tuple[str, ...]] = (
    "the term label",
    "the site code",
    "the column name",
    "the most frequent data value",
)

_PREFIX_CANON: Final[dict[str, str]] = {
    e.curie_prefix.lower(): e.curie_prefix for e in ONTOLOGY_REGISTRY
}
_BY_ID: Final[dict[str, OntologyEntry]] = {e.id: e for e in ONTOLOGY_REGISTRY}


def entry(ontology_id: str) -> OntologyEntry:
    return _BY_ID[ontology_id.lower()]


def prefix_of(curie: str) -> str:
    """The canonical CURIE prefix (``NCBITaxon`` keeps its case, others are upper-cased), or ``''``."""
    if ":" not in curie:
        return ""
    raw = curie.split(":", 1)[0]
    return _PREFIX_CANON.get(raw.lower(), raw.upper())


def allowed_for_aspect(aspect: Aspect) -> frozenset[str]:
    """Ontology ids that may serve an aspect; ``other`` allows everything."""
    if aspect == "other":
        return frozenset(_BY_ID)
    return frozenset(e.id for e in ONTOLOGY_REGISTRY if aspect in e.aspects)


def mask_for_aspect(aspect: Aspect, in_play: frozenset[str] | None = None) -> np.ndarray:
    """Boolean mask over ``ONTOLOGY_OPTIONS`` in registry order (used after the decision,
    the 2048 pattern: mask, renormalise, log the masked mass)."""
    allowed = allowed_for_aspect(aspect)
    if in_play is not None:
        allowed = allowed & in_play
    return np.array([e.id in allowed for e in ONTOLOGY_REGISTRY], dtype=bool)


# -- DataCite controlled vocabularies (DataCite Metadata Schema 4.x), frozen here so the question
# option lists never move; ``tests/test_registry.py`` asserts parity with
# ``mesa_mcp.datacite.schema`` (M5). ResourceTypeGeneral has 28 members, above AnyJev's 26-option
# cap, so it is asked as a yes/no twin per member; the others get a choice plus a twin.
DATACITE_RESOURCE_TYPES: Final[tuple[str, ...]] = (
    "Audiovisual",
    "Book",
    "BookChapter",
    "Collection",
    "ComputationalNotebook",
    "ConferencePaper",
    "ConferenceProceeding",
    "DataPaper",
    "Dataset",
    "Dissertation",
    "Event",
    "Image",
    "InteractiveResource",
    "Journal",
    "JournalArticle",
    "Model",
    "OutputManagementPlan",
    "PeerReview",
    "PhysicalObject",
    "Preprint",
    "Report",
    "Service",
    "Software",
    "Sound",
    "Standard",
    "Text",
    "Workflow",
    "Other",
)
DATACITE_CONTRIBUTOR_TYPES: Final[tuple[str, ...]] = (
    "ContactPerson",
    "DataCollector",
    "DataCurator",
    "DataManager",
    "Distributor",
    "Editor",
    "HostingInstitution",
    "Producer",
    "ProjectLeader",
    "ProjectManager",
    "ProjectMember",
    "RegistrationAgency",
    "RegistrationAuthority",
    "RelatedPerson",
    "Researcher",
    "ResearchGroup",
    "RightsHolder",
    "Sponsor",
    "Supervisor",
    "WorkPackageLeader",
    "Other",
)
DATACITE_RELATION_TYPES: Final[tuple[str, ...]] = (
    "IsCitedBy",
    "Cites",
    "IsSupplementTo",
    "IsSupplementedBy",
    "IsContinuedBy",
    "Continues",
    "IsDescribedBy",
    "Describes",
    "IsPartOf",
    "HasPart",
    "IsReferencedBy",
    "References",
    "IsDocumentedBy",
    "Documents",
    "IsCompiledBy",
    "Compiles",
    "IsVariantFormOf",
    "IsDerivedFrom",
    "IsSourceOf",
    "IsVersionOf",
    "HasVersion",
    "IsNewVersionOf",
    "IsObsoletedBy",
)
DATACITE_DATE_TYPES: Final[tuple[str, ...]] = (
    "Accepted",
    "Available",
    "Copyrighted",
    "Collected",
    "Created",
    "Issued",
    "Submitted",
    "Updated",
    "Valid",
    "Withdrawn",
)
DATACITE_DESCRIPTION_TYPES: Final[tuple[str, ...]] = (
    "Abstract",
    "Methods",
    "SeriesInformation",
    "TableOfContents",
    "TechnicalInfo",
    "Other",
)
DATACITE_VOCABULARIES: Final[dict[str, tuple[str, ...]]] = {
    "ResourceTypeGeneral": DATACITE_RESOURCE_TYPES,
    "ContributorType": DATACITE_CONTRIBUTOR_TYPES,
    "RelationType": DATACITE_RELATION_TYPES,
    "DateType": DATACITE_DATE_TYPES,
    "DescriptionType": DATACITE_DESCRIPTION_TYPES,
}
