document.addEventListener('DOMContentLoaded', function() {
    // Get the current URL's path (e.g., '/', '/classified')
    const currentPath = window.location.pathname;

    // Select all the navigation links
    const navLinks = document.querySelectorAll('.navbar-center-group .nav-link');

    // Loop through each link to check if it matches the current page
    navLinks.forEach(link => {
        // Get the path from the link's href attribute
        const linkPath = new URL(link.href).pathname;
        
        // If the link's path matches the current page's path, add the 'active' class
        if (linkPath === currentPath) {
            link.classList.add('active');
        }
    });
});