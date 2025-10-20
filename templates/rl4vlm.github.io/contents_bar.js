var dContents = document.querySelector('d-contents');
var dArticle = document.querySelector('d-article');
// Get the computed style of the element to access the margin
var computedStyle = window.getComputedStyle(dContents);
// Get the top margin as an integer
var marginTop = parseInt(computedStyle.marginTop, 10);
// Calculate the original top offset plus the margin-top
var originalOffsetTop = dContents.offsetTop;
var originalOffsetLeft = dContents.offsetLeft;
var originalWidth = dContents.offsetWidth; // This should include padding if box-sizing is border-box

// Function to handle the resize event
function onResize() {
    // Recalculate original left and width on resize
    originalOffsetLeft = dContents.offsetLeft;
    originalWidth = dContents.offsetWidth; // This should include padding if box-sizing is border-box
}

// Add the resize event listener
window.addEventListener('resize', onResize);

window.addEventListener('scroll', function() {
    var scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
    var dArticleBottom = dArticle.offsetTop + dArticle.offsetHeight;
    var dContentsActualTop = scrollPosition > originalOffsetTop ? scrollPosition : originalOffsetTop;
    var dContentsBottom = dContentsActualTop + dContents.offsetHeight;
    console.log("dArticleBottom", dArticleBottom)
    console.log("dContentsBottom", dContentsBottom)
    if (dContentsBottom >= dArticleBottom) {
        // Make d-contents invisible
        dContents.style.visibility = 'hidden';
    } else {
        // Make d-contents visible
        dContents.style.visibility = 'visible';
    }

    // Adjust the condition to account for margin-top
    if (scrollPosition + marginTop >= originalOffsetTop) {
        dContents.style.position = 'fixed';
        dContents.style.top = '0px';
        dContents.style.left = originalOffsetLeft + 'px'; // Maintain the original horizontal position
        dContents.style.width = originalWidth + 'px'; // Maintain the original width
    } else {
        dContents.style.position = '';
        dContents.style.top = '';
        dContents.style.left = '';
        dContents.style.width = ''; // Allow the width to be automatic
    }

    
});


// Function to determine which section is in view
function getActiveSection() {
    var sections = document.querySelectorAll('section'); // Assuming your sections have a 'section' tag
    var scrollPosition = window.pageYOffset || document.documentElement.scrollTop;

    for (var i = 0; i < sections.length; i++) {
        if (sections[i].offsetTop <= scrollPosition && sections[i].offsetTop + sections[i].offsetHeight > scrollPosition) {
            return sections[i].id;
        }
    }
    return null;
}

// Function to update the navigation items
function updateNavigation() {
    var activeSection = getActiveSection();
    var navLinks = document.querySelectorAll('d-contents nav a');

    navLinks.forEach(function(navLink) {
        if (navLink.getAttribute('href') === '#' + activeSection) {
            navLink.classList.add('active-nav-item');
        } else {
            navLink.classList.remove('active-nav-item');
        }
    });
}

// Add the scroll event listener
window.addEventListener('scroll', updateNavigation);

// Initialize width and position
onResize();
// Initial update
updateNavigation();