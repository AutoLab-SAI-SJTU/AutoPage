function currentSlidelong(sliderIndex, slideIndex) {
    const sliderlong = document.getElementById(`sliderlong${sliderIndex}`);
    const slideslong = sliderlong.getElementsByClassName('slidelong');
    const dotslong = sliderlong.getElementsByClassName('dotlong');

    for (let i = 0; i < slideslong.length; i++) {
        slideslong[i].style.display = 'none';
        dotslong[i].classList.remove('active');
    }

    slideslong[slideIndex - 1].style.display = 'block';
    dotslong[slideIndex - 1].classList.add('active');
}