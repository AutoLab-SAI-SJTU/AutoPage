function currentSlide(sliderIndex, slideIndex) {
    const slider = document.getElementById(`slider${sliderIndex}`);
    const slides = slider.getElementsByClassName('slide');
    const dots = slider.getElementsByClassName('dot');

    for (let i = 0; i < slides.length; i++) {
        slides[i].style.display = 'none';
        dots[i].classList.remove('active');
    }

    slides[slideIndex - 1].style.display = 'block';
    dots[slideIndex - 1].classList.add('active');
}