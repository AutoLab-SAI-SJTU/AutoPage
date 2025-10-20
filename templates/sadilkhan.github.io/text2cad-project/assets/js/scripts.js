document.addEventListener("DOMContentLoaded", function () {
    const tocContainer = document.getElementById("toc-container");
    const tocLinks = document.querySelectorAll("#toc a");
    const sections = Array.from(tocLinks).map(link => document.querySelector(link.getAttribute("href")));

    // Show TOC container
    tocContainer.classList.add("visible");

    // Add smooth scroll behavior to TOC links
    tocLinks.forEach(link => {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            const targetId = link.getAttribute("href").substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 50, // Adjust for fixed header height
                    behavior: "smooth"
                });
            }
        });
    });

    // Function to handle active TOC updates on scroll
    const updateActiveTocLink = () => {
        let currentActiveIndex = -1;

        sections.forEach((section, index) => {
            const sectionTop = section.getBoundingClientRect().top + window.scrollY - 50;
            if (window.scrollY >= sectionTop) {
                currentActiveIndex = index;
            }
        });

        // Update active class
        tocLinks.forEach((link, index) => {
            if (index === currentActiveIndex) {
                link.classList.add("active");
            } else {
                link.classList.remove("active");
            }
        });
    };

    // Listen for scroll events
    window.addEventListener("scroll", updateActiveTocLink);

    // Initial call to highlight the correct TOC item
    updateActiveTocLink();
});





document.addEventListener("DOMContentLoaded", function() {
    let isDarkMode = false
    // Function to toggle dark mode
    function toggleDarkMode() {
        document.body.classList.toggle("dark-mode");
        let arch_image = document.getElementById("arch_image");
        let logo_label = document.getElementById("Lab Logo");
        // let teaser_image = document.getElementById("teaser_image");
        let data_annot= document.getElementById("data_annot");
        let qual_1_image = document.getElementById("qual_1_image");
        let qual_2_image=document.getElementById("qual_2_image");
        let qual_3_image = document.getElementById("qual_3_image");

        if (!isDarkMode) {
            arch_image.src = "assets/img/arch_dark.png";
            logo_label.src = "assets/img/logo_lab_dark.png";
            // teaser_image.src= "assets/img/teaser_dark.png";
            data_annot.src= "assets/img/data_annot_dark.png";
            qual_1_image.src= "assets/img/qual_1_dark.png";
            qual_2_image.src="assets/img/qual_2_dark.png";
            qual_3_image.src="assets/img/qual_3_dark.svg";

            isDarkMode = true
        } else {
            arch_image.src = "assets/img/arch_light.png";
            logo_label.src = "assets/img/logo_lab_light.png";
            // teaser_image.src= "assets/img/teaser_light.png"
            data_annot.src= "assets/img/data_annot_light.png";
            qual_1_image.src= "assets/img/qual_1_light.png";
            qual_2_image.src="assets/img/qual_2_light.png";
            qual_3_image.src="assets/img/qual_3_light.svg";
            isDarkMode = false
        }
        

    }

    // Event listener for dark mode toggle switch
    var darkModeToggle = document.getElementById("dark-mode-toggle");

    if (darkModeToggle) {
        darkModeToggle.addEventListener("change", toggleDarkMode);

    } 
    
});

// Function to copy citation metadata to clipboard
// function copyToClipboard() {
//     const citationMetadata = document.querySelector('.citation_metadata code');
//     const range = document.createRange();
//     range.selectNode(citationMetadata);
//     window.getSelection().removeAllRanges();
//     window.getSelection().addRange(range);
//     document.execCommand('copy');
//     window.getSelection().removeAllRanges();
// }

// Event listener for the copy button
// const copyButton = document.getElementById('copyButton');
// if (copyButton) {
//     copyButton.addEventListener('click', copyToClipboard);
// }


document.addEventListener('DOMContentLoaded', function() {
    const title = document.querySelector('.more-research-title');
    const list = document.querySelector('.research-list');

    title.addEventListener('click', function() {
        this.classList.toggle('active');
        list.classList.toggle('active');
    });
});


// document.getElementById('copyButton').addEventListener('click', function() {
//     var button = this;
//     var codeContent = document.getElementById('metadata').innerText.trim();

//     // Attempt to copy using the Clipboard API
//     navigator.clipboard.writeText(codeContent).then(function() {
//         // Success: Change to check mark emoji and animate
//         button.textContent = '✅';
//         button.classList.add('success', 'animate');

//         // Reset back to clipboard emoji after a delay
//         setTimeout(function() {
//             button.textContent = '📋';
//             button.classList.remove('success', 'animate');
//         }, 2000);
//     }, function(err) {
//         // Error: Change to cross mark emoji and animate
//         button.textContent = '❌';
//         button.classList.add('error', 'animate');

//         // Reset back to clipboard emoji after a delay
//         setTimeout(function() {
//             button.textContent = '📋';
//             button.classList.remove('error', 'animate');
//         }, 2000);
//     });
// }

// );


function copyText(elementId, buttonId) {
    const textToCopy = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(textToCopy).then(() => {
        // Change the button text to a tick mark
        const button = document.getElementById(buttonId);
        button.innerHTML = "✅ Copied!";
        button.disabled = true; // Disable the button temporarily

        // Revert back to original text after 3 seconds
        setTimeout(() => {
            button.innerHTML = "📋";
            button.disabled = false;
        }, 1000);
    }).catch(err => {
        console.error('Error copying text: ', err);
    });
}

