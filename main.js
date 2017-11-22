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
function switch_filter(selection) {

}


// function to switch size mapping of scatter plot points
// parameters
// -------------------------------------------------------
// to : int { 0, 1, 2 } : default: 0
function switch_size_mapping(to) {

}


$(document).ready(function() {
    // setup topic selectors

    // load jokes
    $.getJSON('data/reddit_jokes.json', function(data) {
        data.forEach(function(d) {
            joke_set.push([d.score, d.title, d.body, d.id])
        })

        $('#jokes_table').DataTable({
            data: joke_set,
            columns: [
                { title: "" },
                { title: "title" },
                { title: "body" }
            ]
        })
    });
})