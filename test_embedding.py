import cv2
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from insightface.app import FaceAnalysis


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
# This is temporary storage.
# Data will be lost when the Streamlit app restarts.
# =========================================================

if "people" not in st.session_state:
    st.session_state.people = []


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
# FIND BEST MATCH
# =========================================================

def find_best_match(query_embedding):

    # No registered people
    if len(st.session_state.people) == 0:

        return None, None

    best_person = None
    best_score = -1

    # -----------------------------------------------------
    # Compare query against every registered person
    # -----------------------------------------------------

    for person in st.session_state.people:

        stored_embedding = person["embedding"]

        score = float(np.dot(
            query_embedding,
            stored_embedding
        ))

        # Keep highest similarity
        if score > best_score:

            best_score = score
            best_person = person

    return best_person, best_score


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
                # Create person record
                # -------------------------------------------------

                person = {

                    "name": name,

                    "age": age,

                    "address": address,

                    "embedding": vector
                }

                # -------------------------------------------------
                # Store person
                # -------------------------------------------------

                st.session_state.people.append(
                    person
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

        elif len(st.session_state.people) == 0:

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

if len(st.session_state.people) == 0:

    st.info(
        "No people registered yet."
    )

else:

    for i, person in enumerate(
        st.session_state.people
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

            st.write(
                f"**Embedding size:** "
                f"{len(person['embedding'])}"
            )


# =========================================================
# CLEAR DATABASE
# =========================================================

st.divider()

if st.button(
    "🗑️ Clear All Registered People"
):

    st.session_state.people = []

    st.success(
        "All registered people have been removed."
    )

    st.rerun()