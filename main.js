// GLOBAL VARIABLES
joke_set = []
words_or_jokes = 0

// function to switch topics
// parameters
// ---------------------------------------------------------
// position: int {0, 1} : the 'first' or 'second' topic
// to      : int { topic_ids } : the id of the selected topic
function switch_topic(position, to) {

}


// function to switch between words and jokes
// parameters
// -------------------------------------------------------
// selection : int {0 <- words, 1 <- jokes} : default: 0
function switch_word_joke_filter(selection) {
    $('.word_joke_selector').removeClass('active')
    // configure for words
    if (selection === 0) {
        $('#word_selector').addClass('active')        
    }

    // configure for jokes
    else if (selection === 1) {
        $('#joke_selector').addClass('active')  
    }
}


// function to switch size mapping of scatter plot points
// parameters
// -------------------------------------------------------
// to : int { 0, 1, 2 } : default: 0
function switch_size_mapping(to) {

}


$(document).ready(function() {
    // setup selectors
    

    // load jokes
    $.getJSON('data/reddit_jokes.json', function(data) {
        
        // put the jokes in a global array
        data.forEach(function(d) {
            joke_set.push([d.score, d.title, d.body, d.id])
        })

        // setup up the jokes table
        $('#jokes_table').DataTable({
            data: joke_set,
            columns: [
                { title: "" },
                { title: "title" },
                { title: "body" }
            ]
        })
        
        // make the search input look good
        search_bar = $('.dataTables_filter input[type="search"]')
            .addClass('text_input')
        $('.dataTables_filter').attr('style', 'width:100%;text-align:center')
    
        $('.dataTables_length').remove()
    });
})