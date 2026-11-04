"""Visual styling, deliberately separate from simulation and teaching logic."""

from pathlib import Path

import streamlit as st


def render_theme():
    css = Path(__file__).resolve().parent.parent / "assets" / "glass.css"
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.markdown("""
    <div class="lab-hero">
        <div class="eyebrow">THE REGRESSION STUDIO / VIRTUAL LAB</div>
        <h1>Small changes.<br>Big learning moments.</h1>
        <p>Build a line. Bend a curve. Find the balance.<br>
        Explore what your model learns — and how well it handles the unseen.</p>
        <div class="hero-tags"><span>01 · Explore</span><span>02 · Compare</span><span>03 · Understand</span></div>
    </div>
    """, unsafe_allow_html=True)
