MODEL_BUNDLE_VERSION = 1


def make_model_bundle(classifier, embedding_profile: str) -> dict:
    return {
        "version": MODEL_BUNDLE_VERSION,
        "embedding_profile": embedding_profile,
        "classifier": classifier,
    }


def unpack_model_bundle(payload, expected_embedding_profile: str):
    if not isinstance(payload, dict):
        return None, "classifier file does not include embedding profile metadata"

    if payload.get("version") != MODEL_BUNDLE_VERSION:
        return None, "classifier file has an unsupported bundle version"

    actual_embedding_profile = payload.get("embedding_profile")
    if actual_embedding_profile != expected_embedding_profile:
        return (
            None,
            (
                "classifier embedding profile mismatch "
                f"(found={actual_embedding_profile}, expected={expected_embedding_profile})"
            ),
        )

    classifier = payload.get("classifier")
    if classifier is None:
        return None, "classifier bundle does not contain a classifier"

    return classifier, None
