// $(function(){

// });

// $(document).ready(function(){

// });

jQuery(document).ready(function ($) {

    // Setup galleries
    document.getElementById('links1').onclick = function (event) {
        event = event || window.event;
        var target = event.target || event.srcElement,
            link = target.src ? target.parentNode : target,
            options = {
                index: link,
                event: event
            },
            links = this.getElementsByTagName('a');
        blueimp.Gallery(links, options);
    };

    document.getElementById('links2').onclick = function (event) {
        event = event || window.event;
        var target = event.target || event.srcElement,
            link = target.src ? target.parentNode : target,
            options = {
                index: link,
                event: event
            },
            links = this.getElementsByTagName('a');
        blueimp.Gallery(links, options);
    };

    document.getElementById('links3').onclick = function (event) {
        event = event || window.event;
        var target = event.target || event.srcElement,
            link = target.src ? target.parentNode : target,
            options = {
                index: link,
                event: event
            },
            links = this.getElementsByTagName('a');
        blueimp.Gallery(links, options);
    };




    // /*======= Skillset *=======*/


    // $('.level-bar-inner').css('width', '0');

    // $(window).on('load', function () {

    //     $('.level-bar-inner').each(function () {

    //         var itemWidth = $(this).data('level');

    //         $(this).animate({
    //             width: itemWidth
    //         }, 800);

    //     });

    // });



});