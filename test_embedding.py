import cv2
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from insightface.app import FaceAnalysis

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


# =========================================================
# PAGE CONFIG — must be the first Streamlit call
# =========================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="centered"
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()


# =========================================================
# SESSION STORAGE
# =========================================================
# st.session_state.vectorstore holds the actual FAISS index
# (name + embedding + metadata) used for matching.
# st.session_state.roster is a lightweight parallel list
# (name/age/address only) used just to render the
# "Registered People" list without reaching into FAISS
# internals.
# Both are still session-only — restarting the app clears
# them, same as before.
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "roster" not in st.session_state:
    st.session_state.roster = []


# =========================================================
# LOAD FACE MODEL
# =========================================================

@st.cache_resource
def load_face_model():

    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    model.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return model


face_model = load_face_model()


# =========================================================
# LANGCHAIN EMBEDDINGS WRAPPER
# =========================================================
# LangChain's Embeddings interface (embed_documents/embed_query)
# is built around embedding *text*. Our embeddings come from
# images via ArcFace, so this class exists only to satisfy
# FAISS's constructor, which requires an Embeddings instance.
# Actual vectors always come from generate_embedding() below,
# and we add/query the FAISS index at the vector level
# (add_embeddings / similarity_search_with_score_by_vector),
# so embed_documents/embed_query are never actually called.
# =========================================================

class ArcFaceEmbeddings(Embeddings):

    def embed_documents(self, texts):
        raise NotImplementedError(
            "ArcFace embeddings come from generate_embedding() on "
            "an image, not from text — use add_embeddings() instead."
        )

    def embed_query(self, text):
        raise NotImplementedError(
            "ArcFace embeddings come from generate_embedding() on "
            "an image, not from text — use similarity_search_by_vector() instead."
        )


face_embeddings = ArcFaceEmbeddings()


# =========================================================
# GENERATE FACE EMBEDDING
# =========================================================

def generate_embedding(image):

    # -----------------------------------------------------
    # Streamlit UploadedFile -> bytes
    # -----------------------------------------------------

    image_bytes = image.getvalue()

    # -----------------------------------------------------
    # Bytes -> NumPy array
    # -----------------------------------------------------

    image_array = np.asarray(
        bytearray(image_bytes),
        dtype=np.uint8
    )

    # -----------------------------------------------------
    # NumPy array -> OpenCV image
    # -----------------------------------------------------

    img = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise ValueError(
            "Could not read the image."
        )

    # -----------------------------------------------------
    # Detect faces
    # -----------------------------------------------------

    faces = face_model.get(img)

    if len(faces) == 0:

        raise ValueError(
            "No face detected in the image."
        )

    if len(faces) > 1:

        raise ValueError(
            "Multiple faces detected. "
            "Please upload an image containing only one face."
        )

    # -----------------------------------------------------
    # Get face embedding
    # -----------------------------------------------------

    embedding = faces[0].embedding

    # -----------------------------------------------------
    # Normalize embedding
    # -----------------------------------------------------

    embedding = embedding / np.linalg.norm(
        embedding
    )

    return embedding


# =========================================================
# STORE A PERSON IN THE VECTORSTORE
# =========================================================

def add_person_to_store(name, age, address, vector):

    metadata = {
        "name": name,
        "age": age,
        "address": address,
    }

    # -----------------------------------------------------
    # distance_strategy=MAX_INNER_PRODUCT is the key detail:
    # FAISS defaults to Euclidean (L2) distance, which is NOT
    # the same number as cosine similarity, even though the
    # ranking it produces happens to agree for unit vectors.
    # Since generate_embedding() already L2-normalizes the
    # vector, inner product == cosine similarity, so this
    # setting makes the FAISS score line up exactly with the
    # 0.5 threshold you were already using.
    # -----------------------------------------------------

    if st.session_state.vectorstore is None:

        st.session_state.vectorstore = FAISS.from_embeddings(
            text_embeddings=[(name, vector.tolist())],
            embedding=face_embeddings,
            metadatas=[metadata],
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )

    else:

        st.session_state.vectorstore.add_embeddings(
            text_embeddings=[(name, vector.tolist())],
            metadatas=[metadata],
        )

    st.session_state.roster.append(metadata)


# =========================================================
# FIND BEST MATCH
# =========================================================

def find_best_match(query_embedding):

    # No registered people
    if st.session_state.vectorstore is None:

        return None, None

    # -----------------------------------------------------
    # Vector-level query — bypasses ArcFaceEmbeddings.embed_query
    # entirely, since we already have the vector.
    # -----------------------------------------------------

    results = st.session_state.vectorstore.similarity_search_with_score_by_vector(
        query_embedding.tolist(),
        k=1,
    )

    if not results:

        return None, None

    doc, score = results[0]

    return doc.metadata, float(score)


# =========================================================
# TITLE
# =========================================================

st.title("👤 Face Recognition System")

st.write(
    "Register people and identify them using "
    "face embeddings and cosine similarity."
)


# =========================================================
# TABS
# =========================================================

add_tab, scan_tab = st.tabs(
    [
        "➕ Add Person",
        "🔍 Scan Face"
    ]
)


# =========================================================
# ADD PERSON TAB
# =========================================================

with add_tab:

    st.header("Add Person")

    # -----------------------------------------------------
    # Upload image
    # -----------------------------------------------------

    uploaded_image = st.file_uploader(
        "Upload person's photo",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="add_image"
    )

    # -----------------------------------------------------
    # Person details
    # -----------------------------------------------------

    name = st.text_input(
        "Name",
        key="person_name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=18,
        step=1,
        key="person_age"
    )

    address = st.text_area(
        "Address",
        key="person_address"
    )

    # -----------------------------------------------------
    # Display uploaded image
    # -----------------------------------------------------

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded Photo",
            width=300
        )

    # -----------------------------------------------------
    # Add person button
    # -----------------------------------------------------

    if st.button(
        "➕ Add Person",
        type="primary"
    ):

        # Validate image

        if uploaded_image is None:

            st.error(
                "Please upload a photo."
            )

        # Validate name

        elif not name.strip():

            st.error(
                "Please enter the person's name."
            )

        else:

            try:

                # -------------------------------------------------
                # Generate face embedding
                # -------------------------------------------------

                with st.spinner("Generating face embedding..."):
                    vector = generate_embedding(
                        uploaded_image
                    )

                # -------------------------------------------------
                # Check for duplicates
                # -------------------------------------------------
                
                existing_person, existing_score = find_best_match(vector)
                
                # Using the same threshold (0.5) used for scanning
                if existing_person is not None and existing_score >= 0.5:
                    st.error(
                        f"❌ Duplicate detected! This face matches "
                        f"**{existing_person['name']}** (Score: {existing_score:.4f})."
                    )
                else:
                    # -------------------------------------------------
                    # Store person in the FAISS vectorstore
                    # -------------------------------------------------

                    add_person_to_store(
                        name,
                        age,
                        address,
                        vector,
                    )

                    st.success(
                        f"✅ {name} added successfully!"
                    )

            except Exception as e:

                st.error(
                    str(e)
                )


# =========================================================
# SCAN FACE TAB
# =========================================================

with scan_tab:

    st.header("Scan Face")

    # -----------------------------------------------------
    # Upload scan image
    # -----------------------------------------------------

    scan_image = st.file_uploader(
        "Upload a photo to identify",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="scan_image"
    )

    # -----------------------------------------------------
    # Display scan image
    # -----------------------------------------------------

    if scan_image:

        st.image(
            scan_image,
            caption="Image to Scan",
            width=300
        )

    # -----------------------------------------------------
    # Scan button
    # -----------------------------------------------------

    if st.button(
        "🔍 Scan",
        type="primary"
    ):

        # -------------------------------------------------
        # Check image
        # -------------------------------------------------

        if scan_image is None:

            st.error(
                "Please upload an image first."
            )

        # -------------------------------------------------
        # Check database
        # -------------------------------------------------

        elif st.session_state.vectorstore is None:

            st.warning(
                "No people have been registered yet."
            )

        else:

            try:

                # -------------------------------------------------
                # Generate embedding for scanned face
                # -------------------------------------------------

                with st.spinner("Scanning face..."):
                    query_vector = generate_embedding(
                        scan_image
                    )

                # -------------------------------------------------
                # Find closest person
                # -------------------------------------------------

                person, score = find_best_match(
                    query_vector
                )

                # -------------------------------------------------
                # Display similarity score
                # -------------------------------------------------

                st.subheader(
                    "Recognition Result"
                )

                st.write(
                    f"Cosine Similarity: "
                    f"**{score:.4f}**"
                )

                # -------------------------------------------------
                # Threshold
                # -------------------------------------------------

                # This is ONLY for experimentation.
                # The correct threshold must be calibrated
                # for your model and dataset.

                threshold = 0.5

                # -------------------------------------------------
                # Match found
                # -------------------------------------------------

                if score >= threshold:

                    st.success(
                        "✅ Person Found"
                    )

                    st.write(
                        f"**Name:** {person['name']}"
                    )

                    st.write(
                        f"**Age:** {person['age']}"
                    )

                    st.write(
                        f"**Address:** {person['address']}"
                    )

                # -------------------------------------------------
                # Unknown person
                # -------------------------------------------------

                else:

                    st.error(
                        "❌ Unknown Person"
                    )

                    st.write(
                        "The similarity score is below "
                        "the recognition threshold."
                    )

            except Exception as e:

                st.error(
                    str(e)
                )


# =========================================================
# REGISTERED PEOPLE
# =========================================================

st.divider()

st.subheader(
    "👥 Registered People"
)

if len(st.session_state.roster) == 0:

    st.info(
        "No people registered yet."
    )

else:

    for i, person in enumerate(
        st.session_state.roster
    ):

        with st.expander(
            f"{i + 1}. {person['name']}"
        ):

            st.write(
                f"**Name:** {person['name']}"
            )

            st.write(
                f"**Age:** {person['age']}"
            )

            st.write(
                f"**Address:** {person['address']}"
            )


# =========================================================
# CLEAR DATABASE
# =========================================================

st.divider()

if st.button(
    "🗑️ Clear All Registered People"
):

    st.session_state.vectorstore = None
    st.session_state.roster = []

    st.success(
        "All registered people have been removed."
    )

    st.rerun()