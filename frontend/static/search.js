import {
    MDCRipple
} from '@material/ripple';

const selector = '.mdc-button, .mdc-icon-button, .mdc-card__primary-action';
const ripples = [].map.call(document.querySelectorAll(selector), function(el) {
    return new MDCRipple(el);
});

var tabBar = new mdc.tabBar.MDCTabBar(document.querySelector('.mdc-tab-bar'));
var contentEls = document.querySelectorAll('.content');

tabBar.listen('MDCTabBar:activated', function(event) {
    // Hide currently-active content
    document.querySelector('.content--active').classList.remove('content--active');
    // Show content for newly-activated tab
    contentEls[event.detail.index].classList.add('content--active');
});
