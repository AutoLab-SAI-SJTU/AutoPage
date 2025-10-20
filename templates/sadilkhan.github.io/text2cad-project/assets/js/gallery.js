document.addEventListener('DOMContentLoaded', function() {
    const slidesContainer = document.querySelector('.slides-container');
    const slides = document.querySelectorAll('.gallery-image');
    const totalSlides = slides.length;
    const slideWidth = slides[0].clientWidth;

    let currentIndex = 1; // Start from the first real slide
    let isAnimating = false;
    let isPlaying = true;
    let slideshowInterval;

    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const playPauseButton = document.getElementById('play-pause-button');
    const playPauseIcon = document.getElementById('play-pause-icon');

    function goToSlide(index) {
        slidesContainer.style.transition = 'transform 0.5s ease-in-out';
        slidesContainer.style.transform = `translateX(-${index * slideWidth}px)`;
    }

    function showNextImage() {
        if (isAnimating) return;
        isAnimating = true;
        currentIndex++;
        goToSlide(currentIndex);
        setTimeout(() => {
            if (currentIndex >= totalSlides - 1) {
                // If we've reached the clone of the first slide
                slidesContainer.style.transition = 'none';
                currentIndex = 1; // Reset to the first real slide
                slidesContainer.style.transform = `translateX(-${currentIndex * slideWidth}px)`;
            }
            isAnimating = false;
        }, 500); // Duration matches CSS transition
    }

    function showPrevImage() {
        if (isAnimating) return;
        isAnimating = true;
        currentIndex--;
        goToSlide(currentIndex);
        setTimeout(() => {
            if (currentIndex <= 0) {
                // If we've reached the clone of the last slide
                slidesContainer.style.transition = 'none';
                currentIndex = totalSlides - 2; // Reset to the last real slide
                slidesContainer.style.transform = `translateX(-${currentIndex * slideWidth}px)`;
            }
            isAnimating = false;
        }, 500);
    }

    function startSlideshow() {
        isPlaying = true;
        updatePlayPauseIcon();
        slideshowInterval = setInterval(showNextImage, 6000);
    }

    function stopSlideshow() {
        isPlaying = false;
        updatePlayPauseIcon();
        clearInterval(slideshowInterval);
    }

    function updatePlayPauseIcon() {
        if (isPlaying) {
            playPauseIcon.classList.remove('fa-play');
            playPauseIcon.classList.add('fa-pause');
        } else {
            playPauseIcon.classList.remove('fa-pause');
            playPauseIcon.classList.add('fa-play');
        }
    }

    prevButton.addEventListener('click', function() {
        stopSlideshow();
        showPrevImage();
    });

    nextButton.addEventListener('click', function() {
        stopSlideshow();
        showNextImage();
    });

    playPauseButton.addEventListener('click', function() {
        if (isPlaying) {
            stopSlideshow();
        } else {
            startSlideshow();
        }
    });

    // Initialize the gallery
    slidesContainer.style.transform = `translateX(-${currentIndex * slideWidth}px)`;
    startSlideshow();

    // Handle window resize to get updated slide width
    window.addEventListener('resize', function() {
        const newSlideWidth = slides[0].clientWidth;
        slidesContainer.style.transition = 'none';
        slidesContainer.style.transform = `translateX(-${currentIndex * newSlideWidth}px)`;
    });
});
