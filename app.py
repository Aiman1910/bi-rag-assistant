
# --------------------------------------------------------------- sidebar
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
<div class="brand">
  <div class="logo">BI</div>
  <div>
    <div class="name">Business Intelligence</div>
    <div class="tag">Insights Assistant</div>
  </div>
</div>
<div class="side-label">About</div>
<div class="note">
  Answers are generated from your CSV datasets and PDF reports.
  Every response lists the sources it used.
</div>
<div class="side-label">Conversation</div>
""",
        unsafe_allow_html=True,
    )

    if st.button(
        "Clear conversation",
        icon=":material/delete_sweep:",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()
